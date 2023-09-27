#!/bin/bash

SWAPFILE="/appdata/swapfile"
SWAP_SIZE_GB=30

if [ -f "$SWAPFILE" ]; then
  SWAP_SIZE_MB=$((SWAP_SIZE_GB * 1024))

  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
  swapon "$SWAPFILE"

  echo "Swap file $SWAPFILE created and enabled."
else
  SWAP_SIZE_MB=$((SWAP_SIZE_GB * 1024))
  fallocate -l ${SWAP_SIZE_MB}M "$SWAPFILE"
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
  swapon "$SWAPFILE"

  echo "Swap file $SWAPFILE created and enabled."
fi

tail -f /dev/null &wait




