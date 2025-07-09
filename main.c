#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256

#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }

float dt = 0.3f;
float diff = 0.0001f;
float visc = 0.0f;

float *density;
float *density_prev;

float *vx;
float *vy;
float *vx_prev;
float *vy_prev;

int win_width = 800;
int win_height = 800;

int mouse_x, mouse_y;
int mouse_button = -1;
int mouse_state = -1;

#define IX(i, j) ((i) + (N + 2) * (j))

void add_source(float *x, float *s, float dt) {
  int size = (N + 2) * (N + 2);
  for (int i = 0; i < size; i++) {
    x[i] += s[i] * dt;
  }
}

void set_bnd(int b, float *x) {

  int i;
  for (i = 1; i <= N; i++) {
    x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
    x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
  }
  x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
  x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
  x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

// Gauss Seidel
void lin_solve(int b, float *x, float *x0, float a, float c) {

  for (int k = 0; k < 20; k++) {
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {

        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                           x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                      c;
      }
    }
    set_bnd(b, x);
  }
}

void diffuse(int b, float *x, float *x0, float diff, float dt) {
  int i, j, k;
  float a = dt * diff * N * N;
  for (k = 0; k < 20; k++) {
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= N; j++) {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                           x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                      (1 + 4 * a);
      }
    }
    set_bnd(b, x);
  }
}

void advect(int b, float *d, float *d0, float *velocX, float *velocY,
            float dt) {
  int i, j, i0, j0, i1, j1;
  float x, y, s0, t0, s1, t1, dt0;
  dt0 = dt * N;
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= N; j++) {
      x = i - dt0 * velocX[IX(i, j)];
      y = j - dt0 * velocY[IX(i, j)];
      if (x < 0.5)
        x = 0.5;
      if (x > N + 0.5)
        x = N + 0.5;
      i0 = (int)x;
      i1 = i0 + 1;
      if (y < 0.5)
        y = 0.5;
      if (y > N + 0.5)
        y = N + 0.5;
      j0 = (int)y;
      j1 = j0 + 1;
      s1 = x - i0;
      s0 = 1 - s1;
      t1 = y - j0;
      t0 = 1 - t1;
      d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                    s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
  }
  set_bnd(b, d);
}

void project(float *velocX, float *velocY, float *p, float *div) {
  int i, j, k;
  float h;
  h = 1.0 / N;
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= N; j++) {
      div[IX(i, j)] = -0.5 * h *
                      (velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)] +
                       velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)]);
      p[IX(i, j)] = 0;
    }
  }
  set_bnd(0, div);
  set_bnd(0, p);
  for (k = 0; k < 20; k++) {
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= N; j++) {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                       p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                      4;
      }
    }
    set_bnd(0, p);
  }
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= N; j++) {
      velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
      velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    }
  }
  set_bnd(1, velocX);
  set_bnd(2, velocY);
}

void dens_step(float *x, float *x0, float *velocX, float *velocY, float diff,
               float dt) {
  add_source(x, x0, dt);
  SWAP(x0, x);
  diffuse(0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(0, x, x0, velocX, velocY, dt);
}

void vel_step(float *velocX, float *velocY, float *velocX0, float *velocY0,
              float visc, float dt) {
  add_source(velocX, velocX0, dt);
  add_source(velocY, velocY, dt);
  SWAP(velocX0, velocX);
  diffuse(1, velocX, velocX0, visc, dt);
  SWAP(velocY0, velocY);
  diffuse(2, velocY, velocY0, visc, dt);
  project(velocX, velocY, velocX0, velocY0);
  SWAP(velocX0, velocX);
  SWAP(velocY0, velocY);
  advect(1, velocX, velocX0, velocX0, velocY0, dt);
  advect(2, velocY, velocY0, velocX0, velocY0, dt);
  project(velocX, velocY, velocX0, velocY0);
}

void init_fluid_simulation() {
  int size = (N + 2) * (N + 2);

  density = (float *)calloc(size, sizeof(float));
  density_prev = (float *)calloc(size, sizeof(float));
  vx = (float *)calloc(size, sizeof(float));
  vy = (float *)calloc(size, sizeof(float));
  vx_prev = (float *)calloc(size, sizeof(float));
  vy_prev = (float *)calloc(size, sizeof(float));

  if (!density || !density_prev || !vx || !vy || !vx_prev || !vy_prev) {
    fprintf(stderr,
            "Error: Could not allocate memory for fluid fields. Exiting.\n");
    exit(1);
  }

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  glShadeModel(GL_FLAT);
}

void cleanup() {
  free(density);
  free(density_prev);
  free(vx);
  free(vy);
  free(vx_prev);
  free(vy_prev);
  fprintf(stdout, "Cleaned up memory. Exiting.\n");
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT);

  float cell_size_x = (float)win_width / N;
  float cell_size_y = (float)win_height / N;

  glBegin(GL_QUADS);

  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {

      float d = density[IX(i, j)];

      d = fmaxf(0.0f, fminf(1.0f, d));
      glColor3f(d, d / 2, d / 3);

      float x0 = (i - 1) * cell_size_x;
      float y0 = (j - 1) * cell_size_y;
      float x1 = i * cell_size_x;
      float y1 = j * cell_size_y;

      glVertex2f(x0, y0);
      glVertex2f(x1, y0);
      glVertex2f(x1, y1);
      glVertex2f(x0, y1);
    }
  }
  glEnd();

  glutSwapBuffers();
}

void reshape(int w, int h) {
  win_width = w;
  win_height = h;
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void idle() {

  vel_step(vx, vy, vx_prev, vy_prev, visc, dt);
  dens_step(density, density_prev, vx, vy, diff, dt);

  memset(density_prev, 0, (N + 2) * (N + 2) * sizeof(float));
  memset(vx_prev, 0, (N + 2) * (N + 2) * sizeof(float));
  memset(vy_prev, 0, (N + 2) * (N + 2) * sizeof(float));

  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
  printf("x: %d, y: %d", x, y);
  mouse_x = x;

  mouse_y = win_height - y;
  mouse_button = button;
  mouse_state = state;
}

void motion(int x, int y) {
  int curr_x = x;
  int curr_y = win_height - y;

  int i = (int)(curr_x / ((float)win_width / N)) + 1;
  int j = (int)(curr_y / ((float)win_height / N)) + 1;

  if (i < 1 || i > N || j < 1 || j > N)
    return;

  if (mouse_button == GLUT_LEFT_BUTTON && mouse_state == GLUT_DOWN) {

    density_prev[IX(i, j)] += 10.0f;
  } else if (mouse_button == GLUT_RIGHT_BUTTON && mouse_state == GLUT_DOWN) {

    float dx = (float)(curr_x - mouse_x);
    float dy = (float)(curr_y - mouse_y);

    vx_prev[IX(i, j)] += dx * 5.0f;
    vy_prev[IX(i, j)] += dy * 5.0f;
  }

  mouse_x = curr_x;
  mouse_y = curr_y;
}

void keyboard(unsigned char key, int x, int y) {
  if (key == 27) {
    cleanup();
    exit(0);
  }
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

  glutInitWindowSize(win_width, win_height);

  glutCreateWindow("Simple 2D Fluid Simulation (C + OpenGL)");

  init_fluid_simulation();

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);

  glutMainLoop();

  return 0;
}
