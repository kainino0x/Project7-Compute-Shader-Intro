#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "kernel.h"

//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Texcoords" };
GLuint pbo = (GLuint)NULL;
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[2];

const unsigned int HEIGHT_FIELD = 0;
const unsigned int PASS_THROUGH = 1;

const int field_width  = 800;
const int field_height = 800;

float fovy = 60.0f;
float zNear = 0.10;
float zFar = 5.0;

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(1.75, 1.75, 1.35);

//====================================
// CUDA Stuff
//====================================

int width = 1000;
int height = 1000;

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void runCUDA();

//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);
void initPBO(GLuint *pbo);
void initCUDA();
void initTextures();
void initVAO();
void initShaders(GLuint *program);

//====================================
// Cleanup Stuff
//====================================
void cleanupCUDA();
void deletePBO(GLuint *pbo);
void deleteTexture(GLuint *tex);
