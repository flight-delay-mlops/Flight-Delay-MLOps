#!/usr/bin/env bash
# One-time setup for the project VMs (Ubuntu 24.04 on the DEI cloud).
# Run on the VM with: sudo bash setup_vm.sh
set -e

# Updates (they include a new kernel, hence the reboot at the end)
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y dist-upgrade

# Docker + Docker Compose, with Docker's official install script
curl -fsSL https://get.docker.com | sh
usermod -aG docker admin  # use docker without sudo

# The VM image only uses the first ~25 GB of the disk: grow the partition,
# then the LVM volume and its filesystem, to fill the whole disk
growpart /dev/xvda 3
pvresize /dev/xvda3
lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

reboot
