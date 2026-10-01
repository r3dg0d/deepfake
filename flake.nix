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
        python = pkgs.python312;
        # CUDA/torch are OPTIONAL — default shell stays CPU-friendly so
        # `devices` / `models list` / `--help` work without NVIDIA.
        deepfake = python.pkgs.buildPythonApplication {
          pname = "deepfake";
          version = "0.5.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = with python.pkgs; [ hatchling ];
          nativeCheckInputs = (with python.pkgs; [ pytestCheckHook ]) ++ [ pkgs.ffmpeg ];
          doCheck = true;
          pytestFlags = [ "-m" "not gpu" ];
          # nixpkgs ships OpenCV as `opencv4`, not the PyPI `opencv-python-headless` dist.
          pythonRemoveDeps = [ "opencv-python-headless" ];
          propagatedBuildInputs = with python.pkgs; [
            click
            numpy
            opencv4
            rich
            pillow
            onnxruntime
            onnx
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
            python.pkgs.pillow
            python.pkgs.onnxruntime
            python.pkgs.onnx
            python.pkgs.cryptography
            python.pkgs.pytest
            python.pkgs.ruff
            pkgs.ffmpeg
            pkgs.v4l-utils
            pkgs.openssl
            pkgs.quickshell
          ];
          shellHook = ''
            echo "deepfake dev shell (CPU-friendly). For AlphaFace GPU: install torch+CUDA yourself."
          '';
        };
        devShells.cuda = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.default ];
          shellHook = ''
            # Let the isolated venv choose GPU wheels instead of CPU shell packages.
            unset PYTHONPATH NIX_PYTHONPATH
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.glib pkgs.libglvnd pkgs.libpng pkgs.libjpeg_turbo pkgs.libtiff pkgs.libwebp ]}:''${LD_LIBRARY_PATH:-}"
            echo "CUDA wheel runtime shell: use an isolated venv with compatible torch + .[vision-cuda12,provenance,watermark]."
          '';
        };
      });
}
