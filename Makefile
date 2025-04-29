.PHONY: all clean vst vst3 arm universal validate install-macos install-macos-vst3 help

# Default target builds for the current platform
all:
	cargo build --release

# Build VST2 format for macOS ARM
vst-arm:
	./build_macos_arm.sh

# Build VST3 format for macOS ARM
vst3-arm:
	./build_macos_arm_vst3.sh

# Build both VST2 and VST3 formats for macOS ARM
arm:
	./build_all_macos_arm.sh

# Build universal macOS binary (Intel + ARM)
universal:
	./build_macos_universal.sh

# Validate the VST2 and VST3 bundles for macOS
validate:
	./validate_macos_vst.sh

# Clean all build artifacts
clean:
	cargo clean
	rm -rf dist

# Install the VST2 plugin locally for testing (macOS)
install-macos: all
	mkdir -p ~/Library/Audio/Plug-Ins/VST/
	cp -r target/release/BasicSynthVST.vst ~/Library/Audio/Plug-Ins/VST/
	@echo "VST2 plugin installed to ~/Library/Audio/Plug-Ins/VST/"

# Install the VST3 plugin locally for testing (macOS)
install-macos-vst3: all
	mkdir -p ~/Library/Audio/Plug-Ins/VST3/
	cp -r target/release/BasicSynthVST.vst3 ~/Library/Audio/Plug-Ins/VST3/
	@echo "VST3 plugin installed to ~/Library/Audio/Plug-Ins/VST3/"

# Full build and install for macOS (both VST2 and VST3)
install-macos-all: arm
	mkdir -p ~/Library/Audio/Plug-Ins/VST/
	mkdir -p ~/Library/Audio/Plug-Ins/VST3/
	cp -r target/aarch64-apple-darwin/release/BasicSynthVST.vst ~/Library/Audio/Plug-Ins/VST/
	cp -r target/aarch64-apple-darwin/release/BasicSynthVST.vst3 ~/Library/Audio/Plug-Ins/VST3/
	@echo "VST2 and VST3 plugins installed to ~/Library/Audio/Plug-Ins/ directories"

# Help message
help:
	@echo "Makefile targets:"
	@echo "  all              - Build for current platform (default)"
	@echo "  vst-arm          - Build VST2 format for macOS ARM"
	@echo "  vst3-arm         - Build VST3 format for macOS ARM"
	@echo "  arm              - Build both VST2 and VST3 formats for macOS ARM"
	@echo "  universal        - Build universal macOS binary (Intel + ARM)"
	@echo "  validate         - Validate macOS VST2/VST3 bundles"
	@echo "  clean            - Clean build artifacts"
	@echo "  install-macos    - Install VST2 plugin to user's VST directory"
	@echo "  install-macos-vst3 - Install VST3 plugin to user's VST3 directory"
	@echo "  install-macos-all - Install both VST2 and VST3 plugins"
	@echo "  help             - Show this help message"