{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
      python311
      python311Packages.pip
      python311Packages.virtualenv
    ]);
  runScript = "bash";
}).env
