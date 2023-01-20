#include <stdio.h>
#include "array-print.h"

void array_print(int arr[][5], int rows) {
  int i, j;

  for (i= 0; i < rows; i++) {
    for (j= 0; j < 5; j++)
      printf("%d ", arr[i][j]);
    printf("\n");
  }
}

