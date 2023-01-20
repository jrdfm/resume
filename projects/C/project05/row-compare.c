#include "row-compare.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int row_compare(int arr1[][5], int arr2[][5], int row) {
  int same= 1, i= 0;

  while (i < 5 && same)
    if (arr1[row][i] != arr2[row][i])
      same= 0;
    else i++;

  return same;
}

