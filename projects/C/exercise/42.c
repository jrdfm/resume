#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void *deep_thoughts(void *vargp) {
  
  pthread_exit((void *)42);
}
int main() {
  int i;
  pthread_t tid;
  pthread_create(&tid, NULL, deep_thoughts, NULL);
  pthread_join(tid, (void **)&i);
  printf("%d\n", i);
  return 0;
}