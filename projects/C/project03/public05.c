#include <stdio.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 5 (public05.c)
 *
 * Tests calling disassemble() on a program with some valid instructions.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define PROGRAM_SIZE 5

int main(void) {
  Wrd program[PROGRAM_SIZE]= {0x00298000, 0x084e0000, 0x10728000, 0x68b18000,
                              0x702e8000};
  unsigned short location= -1;

  disassemble(program, PROGRAM_SIZE, &location);

  return 0;
}
