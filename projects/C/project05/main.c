#include <stdio.h>
#include "range-fill.h"
#include "array-print.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  int arr1[][5]= {{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}, {1, 3, 5, 7, 9}};
  int arr2[4][5];

  range_fill(arr2, 0, 0, 4, 1, 123);
  range_fill(arr2, 0, 2, 2, 3, 456);
  range_fill(arr2, 3, 2, 4, 3, 789);

  array_print(arr1, 3);
  printf("\n");
  array_print(arr2, 4);

  return 0;
}
