#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

# Default version bump is patch unless specified (patch, minor, or major)
VERSION_TYPE=${1:-patch}

echo "Bumping version: $VERSION_TYPE"

# Bump the version using bump-my-version.
# This command should update version files, commit the changes, and create a Git tag.
uv bump-my-version bump "$VERSION_TYPE" --verbose

# Push commits and tags to the remote repository.
echo "Pushing changes to remote repository..."
git push --follow-tags

# Retrieve the new version tag (latest tag)
NEW_TAG=$(git describe --tags --abbrev=0)
echo "New version tag: $NEW_TAG"

echo "Release process complete!"