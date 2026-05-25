{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pokemon-showdown = {
      url = "github:billy4479/pokemon-showdown-wrapper";
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
        showdown-wrapper = pokemon-showdown.packages.${system}.default;

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
            plotly
            deap

            jupyterlab

            marimo
            watchdog

            python-lsp-server
            python-lsp-ruff

            showdown-wrapper
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
