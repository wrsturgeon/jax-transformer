{
  description = "Clear, concise, & efficient implementation of a Transformer in pure JAX.";
  inputs = {
    check-and-compile = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixfmt.follows = "nixfmt";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:wrsturgeon/check-and-compile";
    };
    flake-utils.url = "github:numtide/flake-utils";
    jax-attention = {
      inputs = {
        check-and-compile.follows = "check-and-compile";
        flake-utils.follows = "flake-utils";
        nixfmt.follows = "nixfmt";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:wrsturgeon/jax-attention";
    };
    nixfmt = {
      inputs.flake-utils.follows = "flake-utils";
      url = "github:serokell/nixfmt";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      check-and-compile,
      flake-utils,
      jax-attention,
      nixfmt,
      nixpkgs,
      self,
    }:
    let
      pname = "jax-transformer";
      pyname = "jax_transformer";
      version = "0.0.1";
      src = ./.;
      default-pkgs =
        p: py:
        with py;
        [
          beartype
          jaxtyping
        ]
        ++ [
          (check-and-compile.lib.with-pkgs p py)
          (jax.overridePythonAttrs (
            old:
            old
            // {
              doCheck = false;
              propagatedBuildInputs = old.propagatedBuildInputs ++ [ py.jaxlib-bin ];
            }
          ))
          (jax-attention.lib.with-pkgs p py)
        ];
      check-pkgs =
        p: py: with py; [
          hypothesis
          mypy
          pytest
        ];
      ci-pkgs =
        p: py: with py; [
          black
          coverage
        ];
      dev-pkgs =
        p: py: with py; [
          matplotlib
          python-lsp-server
        ];
      lookup-pkg-sets =
        ps: p: py:
        builtins.concatMap (f: f p py) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        pkgs.stdenv.mkDerivation {
          inherit pname version src;
          propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pkgs pypkgs;
          buildPhase = ":";
          installPhase = ''
            mkdir -p $out/${pypkgs.python.sitePackages}
            mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
          '';
        };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps pkgs)}/bin/python";
      in
      {
        packages.ci =
          let
            pname = "ci";
            python = python-with [
              default-pkgs
              check-pkgs
              ci-pkgs
            ];
            find = "${pkgs.findutils}/bin/find";
            nixfmt-bin = "${nixfmt.packages.${system}.default}/bin/nixfmt";
            rm = "${pkgs.coreutils}/bin/rm";
            xargs = "${pkgs.findutils}/bin/xargs";
            exec = ''
              #!${pkgs.bash}/bin/bash

              set -eu

              export JAX_ENABLE_X64=1

              ${rm} -fr result
              ${find} . -name '*.nix' | ${xargs} ${nixfmt-bin} --check
              ${python} -m black --check .
              ${python} -m mypy .

              ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
              ${python} -m coverage report -m --fail-under=100
            '';
          in
          pkgs.stdenv.mkDerivation {
            inherit pname version src;
            buildPhase = ":";
            installPhase = ''
              mkdir -p $out/bin
              echo "${exec}" > $out/bin/${pname}
              chmod +x $out/bin/${pname}
            '';
          };
        devShells.default = pkgs.mkShell {
          JAX_ENABLE_X64 = "1";
          packages = (
            lookup-pkg-sets [
              default-pkgs
              check-pkgs
              ci-pkgs
              dev-pkgs
            ] pkgs pypkgs
          );
        };
      }
    );
}
