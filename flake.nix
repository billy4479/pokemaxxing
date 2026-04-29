{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pokemon-showdown = {
      url = "git+file:../../src/Pokemon-Showdown";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      pokemon-showdown,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        modelEval = pokemon-showdown.packages.${system}.pokemon-showdown-model-eval;

        pyPkgs = pkgs.python3.withPackages (
          python-pkgs: with python-pkgs; [
            numpy
            pandas
            matplotlib
            torch
            tqdm
            playwright
            umap-learn
            scikit-learn

            jupyterlab

            marimo
            watchdog

            python-lsp-server
            python-lsp-ruff
            modelEval
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pyPkgs
            pkgs.nodejs_latest
          ];

          packages = with pkgs; [
            ruff

            marimo
          ];
        };
      }
    );
}
