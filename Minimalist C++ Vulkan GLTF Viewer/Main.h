#include <cstdint>
#pragma once
constexpr uint32_t WIN_W = 800;
constexpr uint32_t WIN_H = 600;

// ---------------------------------------------------------------------------
// Default orientation settings
// Adjust these values to change the initial rotation of the loaded model or
// the camera view. Values are in degrees for convenience.
// ---------------------------------------------------------------------------
constexpr float DEFAULT_MODEL_YAW   = 180.0f; // rotation around Y axis
constexpr float DEFAULT_MODEL_PITCH = 0.0f; // rotation around X axis

constexpr float DEFAULT_CAMERA_YAW   = 0.0f; // rotate camera around Y axis
constexpr float DEFAULT_CAMERA_PITCH = 0.0f; // rotate camera around X axis

// ---------------------------------------------------------------------------
// Default zoom distance
// Controls how far the camera starts from the origin. Higher values mean
// the model appears smaller. Units match the application's world units
// ---------------------------------------------------------------------------
constexpr float DEFAULT_ZOOM = 1.0f;
