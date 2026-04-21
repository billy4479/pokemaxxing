{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pyPkgs = pkgs.python3.withPackages (
          python-pkgs: with python-pkgs; [
            numpy
            pandas
            matplotlib

            jupyterlab

            marimo
            watchdog

            python-lsp-server
            python-lsp-ruff
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pyPkgs
          ];

          packages = with pkgs; [
            ruff

            marimo
          ];
        };
      }
    );
}
