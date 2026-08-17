/* Dependency-free Keccak-256 and ENS namehash implementation. */
(() => {
  'use strict';
  const MASK = (1n << 64n) - 1n;
  const RC = [
    0x0000000000000001n,0x0000000000008082n,0x800000000000808an,0x8000000080008000n,
    0x000000000000808bn,0x0000000080000001n,0x8000000080008081n,0x8000000000008009n,
    0x000000000000008an,0x0000000000000088n,0x0000000080008009n,0x000000008000000an,
    0x000000008000808bn,0x800000000000008bn,0x8000000000008089n,0x8000000000008003n,
    0x8000000000008002n,0x8000000000000080n,0x000000000000800an,0x800000008000000an,
    0x8000000080008081n,0x8000000000008080n,0x0000000080000001n,0x8000000080008008n
  ];
  const ROT = [0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14];
  const rotl = (value, shift) => {
    const n = BigInt(shift % 64);
    if (n === 0n) return value & MASK;
    return ((value << n) | (value >> (64n - n))) & MASK;
  };
  function permute(state) {
    const b = new Array(25).fill(0n);
    const c = new Array(5).fill(0n);
    const d = new Array(5).fill(0n);
    for (let round = 0; round < 24; round += 1) {
      for (let x = 0; x < 5; x += 1) c[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
      for (let x = 0; x < 5; x += 1) d[x] = c[(x+4)%5] ^ rotl(c[(x+1)%5], 1);
      for (let y = 0; y < 5; y += 1) for (let x = 0; x < 5; x += 1) state[x+5*y] = (state[x+5*y] ^ d[x]) & MASK;
      for (let y = 0; y < 5; y += 1) for (let x = 0; x < 5; x += 1) b[y + 5*((2*x + 3*y)%5)] = rotl(state[x+5*y], ROT[x+5*y]);
      for (let y = 0; y < 5; y += 1) for (let x = 0; x < 5; x += 1) state[x+5*y] = (b[x+5*y] ^ ((~b[(x+1)%5+5*y] & MASK) & b[(x+2)%5+5*y])) & MASK;
      state[0] = (state[0] ^ RC[round]) & MASK;
    }
  }
  const bytesToHex = bytes => '0x' + [...bytes].map(v => v.toString(16).padStart(2,'0')).join('');
  const hexToBytes = hex => {
    const clean = String(hex).replace(/^0x/,'');
    if (clean.length % 2) throw new Error('Invalid hexadecimal input');
    const out = new Uint8Array(clean.length / 2);
    for (let i = 0; i < out.length; i += 1) out[i] = parseInt(clean.slice(i*2,i*2+2),16);
    return out;
  };
  function keccak256Bytes(input) {
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    const rate = 136;
    const state = new Array(25).fill(0n);
    let offset = 0;
    while (offset + rate <= bytes.length) {
      for (let i = 0; i < rate; i += 1) state[Math.floor(i/8)] ^= BigInt(bytes[offset+i]) << BigInt(8*(i%8));
      permute(state);
      offset += rate;
    }
    const block = new Uint8Array(rate);
    block.set(bytes.slice(offset));
    block[bytes.length - offset] ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (let i = 0; i < rate; i += 1) state[Math.floor(i/8)] ^= BigInt(block[i]) << BigInt(8*(i%8));
    permute(state);
    const out = new Uint8Array(32);
    for (let i = 0; i < 32; i += 1) out[i] = Number((state[Math.floor(i/8)] >> BigInt(8*(i%8))) & 0xffn);
    return out;
  }
  const keccak256 = input => bytesToHex(keccak256Bytes(typeof input === 'string' ? new TextEncoder().encode(input) : input));
  function namehash(name) {
    let node = new Uint8Array(32);
    const normalized = String(name || '').trim().toLowerCase().replace(/\.$/,'');
    if (!normalized) return bytesToHex(node);
    const labels = normalized.split('.');
    for (let i = labels.length - 1; i >= 0; i -= 1) {
      const labelHash = keccak256Bytes(new TextEncoder().encode(labels[i]));
      const combined = new Uint8Array(64);
      combined.set(node, 0); combined.set(labelHash, 32);
      node = keccak256Bytes(combined);
    }
    return bytesToHex(node);
  }
  const selector = signature => keccak256(signature).slice(0,10);
  window.GoalOSCrypto = Object.freeze({ keccak256, keccak256Bytes, namehash, selector, bytesToHex, hexToBytes });
})();
