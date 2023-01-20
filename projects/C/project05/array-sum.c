#include "array-sum.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int array_sum(int arr[][5], int rows) {
  int i, j, sum= 0;

  for (i= 0; i < rows; i++)
    for (j= 0; j < 5; j++)
      sum += arr[i][j];

  return sum;
}
