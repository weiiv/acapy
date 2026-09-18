"""Tests for external signer dispatch in BaseWallet."""

from typing import Optional
from unittest.mock import AsyncMock

import pytest

from acapy_agent.config.base import InjectionError
from acapy_agent.utils.testing import create_test_profile
from acapy_agent.wallet.askar import AskarWallet
from acapy_agent.wallet.base import BaseWallet
from acapy_agent.wallet.did_info import DIDInfo
from acapy_agent.wallet.did_method import KEY, DIDMethods
from acapy_agent.wallet.error import WalletError, WalletNotFoundError
from acapy_agent.wallet.kanon_wallet import KanonWallet
from acapy_agent.wallet.key_type import ED25519, P256, KeyTypes
from acapy_agent.wallet.signer_registry import Signer, SignerRegistry


class _FakeSession:
    """Minimal session stub exposing `inject_or` for the base wallet."""

    def __init__(self, registry: Optional[SignerRegistry] = None):
        self._registry = registry

    def inject_or(self, cls):
        if cls is SignerRegistry:
            return self._registry
        return None


class _StubWallet(BaseWallet):
    """Minimal wallet stub for `_get_external_signature` tests."""

    LOCAL_SIG = b"local-signature"

    def __init__(self, did_info_by_verkey=None, session=None):
        self._by_verkey = did_info_by_verkey or {}
        self._session = session

    async def get_local_did_for_verkey(self, verkey):
        try:
            return self._by_verkey[verkey]
        except KeyError:
            raise WalletNotFoundError(f"Unknown verkey: {verkey}")

    async def sign_message(self, message, from_verkey):
        # Mirrors the real wallets.
        if (sig := await self._get_external_signature(message, from_verkey)) is not None:
            return sig
        return self.LOCAL_SIG

    @property
    def session(self):
        return self._session

    # --- Trivial stubs to satisfy BaseWallet ABC ---
    async def create_signing_key(self, *a, **kw): ...
    async def create_key(self, *a, **kw): ...
    async def get_signing_key(self, *a, **kw): ...
    async def replace_signing_key_metadata(self, *a, **kw): ...
    async def rotate_did_keypair_start(self, *a, **kw): ...
    async def rotate_did_keypair_apply(self, *a, **kw): ...
    async def create_local_did(self, *a, **kw): ...
    async def store_did(self, *a, **kw): ...
    async def get_public_did(self, *a, **kw): ...
    async def set_public_did(self, *a, **kw): ...
    async def get_local_did(self, *a, **kw): ...
    async def get_local_dids(self, *a, **kw): ...
    async def replace_local_did_metadata(self, *a, **kw): ...
    async def verify_message(self, *a, **kw): ...
    async def pack_message(self, *a, **kw): ...
    async def unpack_message(self, *a, **kw): ...
    async def assign_kid_to_key(self, *a, **kw): ...
    async def get_key_by_kid(self, *a, **kw): ...


def _did_info(verkey: str, metadata: dict) -> DIDInfo:
    return DIDInfo(
        did="did:web:example.com",
        verkey=verkey,
        metadata=metadata,
        method=KEY,
        key_type=P256,
    )


@pytest.mark.asyncio
async def test_signs_locally_when_verkey_unknown():
    """No DID for verkey → no external signer; wallet signs locally."""
    wallet = _StubWallet()
    assert await wallet.sign_message(b"payload", "unknown") == _StubWallet.LOCAL_SIG


@pytest.mark.asyncio
async def test_signs_locally_when_no_signer_marker():
    """DID exists but no `signer` in metadata → wallet signs locally."""
    verkey = "software-verkey"
    wallet = _StubWallet(did_info_by_verkey={verkey: _did_info(verkey, {})})
    assert await wallet.sign_message(b"payload", verkey) == _StubWallet.LOCAL_SIG


@pytest.mark.asyncio
async def test_signs_locally_when_lookup_fails():
    """A DID lookup failure declines rather than breaking signing."""

    class _BrokenWallet(_StubWallet):
        async def get_local_did_for_verkey(self, verkey):
            raise InjectionError("No instance provided for class: KeyTypes")

    wallet = _BrokenWallet()
    assert await wallet.sign_message(b"payload", "any") == _StubWallet.LOCAL_SIG


@pytest.mark.asyncio
async def test_dispatches_to_registered_signer():
    """`signer` marker + registered Signer → external signature is returned."""
    verkey = "hsm-verkey"
    metadata = {"signer": "hsm", "key_ref": "issuer-key-01"}
    registry = SignerRegistry()
    fake_signer = AsyncMock(spec=Signer)
    fake_signer.sign.return_value = b"\x00" * 64  # fake raw r||s for P-256
    registry.register("hsm", fake_signer)

    wallet = _StubWallet(
        did_info_by_verkey={verkey: _did_info(verkey, metadata)},
        session=_FakeSession(registry),
    )

    sig = await wallet.sign_message(b"payload", verkey)

    assert sig == b"\x00" * 64
    fake_signer.sign.assert_awaited_once_with("issuer-key-01", b"payload", P256)


@pytest.mark.asyncio
async def test_raises_when_signer_marker_but_no_registry():
    """Marker present, no SignerRegistry in context → WalletError."""
    verkey = "hsm-verkey"
    metadata = {"signer": "hsm", "key_ref": "issuer-key-01"}
    wallet = _StubWallet(
        did_info_by_verkey={verkey: _did_info(verkey, metadata)},
        session=_FakeSession(registry=None),
    )
    with pytest.raises(WalletError, match=r"requires signer hsm"):
        await wallet.sign_message(b"payload", verkey)


@pytest.mark.asyncio
async def test_raises_when_signer_name_unknown():
    """Unknown signer name → WalletError listing the registered names."""
    verkey = "hsm-verkey"
    metadata = {"signer": "unknown-provider", "key_ref": "issuer-key-01"}
    registry = SignerRegistry()
    registry.register("hsm", AsyncMock(spec=Signer))
    wallet = _StubWallet(
        did_info_by_verkey={verkey: _did_info(verkey, metadata)},
        session=_FakeSession(registry),
    )
    with pytest.raises(WalletError, match=r"registered signers: \['hsm'\]"):
        await wallet.sign_message(b"payload", verkey)


@pytest.mark.asyncio
async def test_raises_when_key_ref_missing():
    """Marker present, registry present, no key_ref → WalletError."""
    verkey = "hsm-verkey"
    metadata = {"signer": "hsm"}  # no key_ref
    registry = SignerRegistry()
    registry.register("hsm", AsyncMock(spec=Signer))
    wallet = _StubWallet(
        did_info_by_verkey={verkey: _did_info(verkey, metadata)},
        session=_FakeSession(registry),
    )
    with pytest.raises(WalletError, match="missing key_ref"):
        await wallet.sign_message(b"payload", verkey)


@pytest.mark.asyncio
async def test_sign_message_remains_abstract():
    """A wallet that omits sign_message cannot be instantiated."""

    class _NoSignMessage(BaseWallet):
        pass

    assert "sign_message" in BaseWallet.__abstractmethods__
    with pytest.raises(TypeError, match="sign_message"):
        _NoSignMessage()


# --- Real wallet delegation ---


async def _askar_profile():
    profile = await create_test_profile()
    profile.context.injector.bind_instance(DIDMethods, DIDMethods())
    profile.context.injector.bind_instance(KeyTypes, KeyTypes())
    return profile


@pytest.mark.asyncio
async def test_askar_wallet_dispatches_to_registered_signer():
    """AskarWallet returns the external signature when the DID delegates."""
    profile = await _askar_profile()
    registry = SignerRegistry()
    fake_signer = AsyncMock(spec=Signer)
    fake_signer.sign.return_value = b"\x03" * 64
    registry.register("hsm", fake_signer)
    profile.context.injector.bind_instance(SignerRegistry, registry)

    async with profile.session() as session:
        wallet = AskarWallet(session)
        did_info = await wallet.create_local_did(
            KEY, P256, metadata={"signer": "hsm", "key_ref": "k1"}
        )
        sig = await wallet.sign_message(b"payload", did_info.verkey)

    assert sig == b"\x03" * 64
    fake_signer.sign.assert_awaited_once_with("k1", b"payload", P256)


@pytest.mark.asyncio
async def test_askar_wallet_signs_locally_without_marker():
    """No marker → AskarWallet signs with its own key and never calls the signer."""
    profile = await _askar_profile()
    registry = SignerRegistry()
    fake_signer = AsyncMock(spec=Signer)
    registry.register("hsm", fake_signer)
    profile.context.injector.bind_instance(SignerRegistry, registry)

    async with profile.session() as session:
        wallet = AskarWallet(session)
        did_info = await wallet.create_local_did(KEY, ED25519)
        sig = await wallet.sign_message(b"payload", did_info.verkey)
        assert await wallet.verify_message(b"payload", sig, did_info.verkey, ED25519)

    fake_signer.sign.assert_not_awaited()


@pytest.mark.asyncio
async def test_kanon_wallet_dispatches_to_registered_signer():
    """KanonWallet returns the external signature when the DID delegates."""
    verkey = "hsm-verkey"
    registry = SignerRegistry()
    fake_signer = AsyncMock(spec=Signer)
    fake_signer.sign.return_value = b"\x04" * 64
    registry.register("hsm", fake_signer)

    wallet = KanonWallet(_FakeSession(registry))
    wallet.get_local_did_for_verkey = AsyncMock(
        return_value=_did_info(verkey, {"signer": "hsm", "key_ref": "k1"})
    )

    assert await wallet.sign_message(b"payload", verkey) == b"\x04" * 64
    fake_signer.sign.assert_awaited_once_with("k1", b"payload", P256)


# --- SignerRegistry unit tests ---


def test_signer_registry_register_and_get():
    reg = SignerRegistry()
    signer = AsyncMock(spec=Signer)
    reg.register("hsm", signer)
    assert reg.get("hsm") is signer
    assert reg.names() == ["hsm"]


def test_signer_registry_get_returns_none_for_unknown():
    reg = SignerRegistry()
    assert reg.get("unknown") is None


def test_signer_registry_rejects_duplicate_registration():
    reg = SignerRegistry()
    reg.register("hsm", AsyncMock(spec=Signer))
    with pytest.raises(ValueError, match="already registered"):
        reg.register("hsm", AsyncMock(spec=Signer))
