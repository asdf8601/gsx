#!/bin/bash
# Release script for manual releases with setuptools_scm
# Usage: ./release.sh 1.2.3

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.2.3"
    exit 1
fi

VERSION=$1

# Validate version format
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.2.3)"
    exit 1
fi

echo "Creating release for version $VERSION"

# Create and push tag (setuptools_scm will handle the version automatically)
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"

echo "Release v$VERSION created successfully!"
echo "setuptools_scm will automatically use the tag version"
echo "Check GitHub Actions for build progress: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"