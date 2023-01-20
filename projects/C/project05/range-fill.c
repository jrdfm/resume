#include "range-fill.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

/* assume coordinates are valid */

void range_fill(int arr[][5], int x1, int y1, int x2, int y2, int value) {
  int i, j;

  for (i= y1; i <= y2; i++)
    for (j= x1; j <= x2; j++)
      arr[i][j]= value;
}
