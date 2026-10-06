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
        # Pinned official CPU wheels avoid a large upstream C++ rebuild and
        # match the 1.22 runtime family validated with this application.
        ortWheels = {
          x86_64-linux = {
            url = "https://files.pythonhosted.org/packages/8c/60/16d219b8868cc8e8e51a68519873bdb9f5f24af080b62e917a13fff9989b/onnxruntime-1.22.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
            hash = "sha256-aWSpdXMa/BncNBj62NTgjEiSAUT/WQFJQppevg0V+zw=";
          };
          aarch64-linux = {
            url = "https://files.pythonhosted.org/packages/03/79/36f910cd9fc96b444b0e728bba14607016079786adf032dae61f7c63b4aa/onnxruntime-1.22.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
            hash = "sha256-yGARKOrvebY2FSrqdq5pgbfJ/IGmGPWEwV141CsxDxw=";
          };
        };
        ortCpu = if builtins.hasAttr system ortWheels then python.pkgs.buildPythonPackage {
          pname = "onnxruntime";
          version = "1.22.0";
          format = "wheel";
          src = pkgs.fetchurl ortWheels.${system};
          nativeBuildInputs = [ pkgs.autoPatchelfHook ];
          buildInputs = [ pkgs.stdenv.cc.cc.lib ];
          pythonRemoveDeps = [ "flatbuffers" "protobuf" "sympy" ];
          dependencies = with python.pkgs; [ numpy coloredlogs packaging ];
          pythonImportsCheck = [ "onnxruntime" ];
          meta.license = pkgs.lib.licenses.mit;
        } else python.pkgs.onnxruntime;
        # CUDA/torch are OPTIONAL — default shell stays CPU-friendly so
        # `devices` / `models list` / `--help` work without NVIDIA.
        deepfake = python.pkgs.buildPythonApplication {
          pname = "deepfake";
          version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
          src = pkgs.lib.cleanSource ./.;
          format = "pyproject";
          nativeBuildInputs = with python.pkgs; [ hatchling ];
          nativeCheckInputs = (with python.pkgs; [ pytestCheckHook ]) ++ [ pkgs.ffmpeg ];
          doCheck = true;
          makeWrapperArgs = [ "--prefix" "PATH" ":" (pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.openssl pkgs.v4l-utils ]) ];
          pythonImportsCheck = [ "deepfake" "deepfake.occlusion" ];
          # nixpkgs ships OpenCV as `opencv4`, not the PyPI `opencv-python-headless` dist.
          pythonRemoveDeps = [ "opencv-python-headless" ];
          propagatedBuildInputs = with python.pkgs; [
            click
            numpy
            opencv4
            rich
            pillow
            ortCpu
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
            ortCpu
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
