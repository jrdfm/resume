#include <stdio.h>
#include "range-fill.h"
#include "array-sum.h"
#include "row-sum.h"

/* (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  int arr1[][5]= {{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}, {1, 3, 5, 7, 9}};
  int arr2[4][5];
  int i;

  range_fill(arr2, 0, 0, 4, 1, 123);
  range_fill(arr2, 0, 2, 2, 3, 456);
  range_fill(arr2, 3, 2, 4, 3, 789);

  printf("%d %d\n", array_sum(arr1, 3), array_sum(arr2, 4));

  for (i= 0; i < 3; i++)
    printf("%d ", row_sum(arr1, i));
  printf("\n");

  for (i= 0; i < 4; i++)
    printf("%d ", row_sum(arr2, i));
  printf("\n");

  return 0;
}
