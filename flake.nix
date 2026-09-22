{
  description = "deepfake — Linux face-swap CLI (AlphaFace research wrapper)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
        # CUDA/torch are OPTIONAL — default shell stays CPU-friendly so
        # `devices` / `models list` / `--help` work without NVIDIA.
        deepfake = python.pkgs.buildPythonApplication {
          pname = "deepfake";
          version = "0.2.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = with python.pkgs; [ hatchling ];
          # nixpkgs ships OpenCV as `opencv4`, not the PyPI `opencv-python-headless` dist.
          pythonRemoveDeps = [ "opencv-python-headless" ];
          propagatedBuildInputs = with python.pkgs; [
            click
            numpy
            opencv4
            rich
          ];
          meta = with pkgs.lib; {
            description = "Linux real-time face-swap CLI (AlphaFace wrapper)";
            license = licenses.mit;
            mainProgram = "deepfake";
            platforms = platforms.linux;
          };
        };
      in {
        packages.default = deepfake;
        packages.deepfake = deepfake;
        apps.default = flake-utils.lib.mkApp { drv = deepfake; };
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            python.pkgs.pip
            python.pkgs.click
            python.pkgs.numpy
            python.pkgs.opencv4
            python.pkgs.rich
            python.pkgs.pytest
            python.pkgs.ruff
            pkgs.ffmpeg
            pkgs.v4l-utils
          ];
          # Optional CUDA tip (do not force):
          #   nix develop .#cuda  # if you add a cuda overlay locally
          shellHook = ''
            echo "deepfake dev shell (CPU-friendly). For AlphaFace GPU: install torch+CUDA yourself."
          '';
        };
      });
}
