# Python environment for the HF reference tools. NixOS has no python on PATH.
#   nix-shell tools/reference/shell.nix --run "python tools/reference/dump_logits.py ..."
# torch from nixpkgs is the CPU build, which is what the reference needs (f32, CPU).
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.torch
      ps.transformers
      ps.safetensors
      ps.numpy
    ]))
  ];
}
