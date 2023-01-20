#include "machine.h"
#include <stdio.h>

/* CMSC 216, Spring 2021, Project #3
 * machine.c
 *
 * Contains functions print_instruction(Wrd instruction),
 * disassemble(const Wrd program[], unsigned short num_instrs,
 * unsigned short *const bad_instr), where_set(const Wrd program[], unsigned
 * short num_words, unsigned short reg_nbr)
 *
 * Yared Fikremariam Section 0302 UID 116945769
 */

unsigned short check_instruction(Wrd instruction);

/*
 * This function extracts and prints instruction if their parameters are valid.
 * It prints the opcode followed by register oprands that are used.
 * Returns 1 if instruction is valid or 0 otherwise.
 *
 */
unsigned short print_instruction(Wrd instruction) {

  Wrd op, r1, r2, r3, memi, ext;
  Op_code ocode;

  op = (((1 << 5) - 1) & (instruction >> (27)));
  ext = (((1 << 3) - 1) & (instruction >> (24)));
  r1 = (((1 << 3) - 1) & (instruction >> (21)));
  r2 = (((1 << 3) - 1) & (instruction >> (18)));
  r3 = (((1 << 3) - 1) & (instruction >> (15)));
  memi = (((1 << 15) - 1) & (instruction >> (0)));

  ocode = op;

  if (op >= 0 || op < 27) {
    switch (ocode) {
    case PLUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "plus");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }

      break;
    case MINUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "minus");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case TIMES:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "times");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case DIV:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "div");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case MOD:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "mod");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case NEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        printf("%-5s", "neg");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
      }
      break;
    case ABS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        printf("%-5s", "abs");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
      }
      break;
    case SHL:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "shl");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case SHR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "shr");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case BAND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "band");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case BOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "bor");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case BXOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "bxor");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case BNEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        printf("%-5s", "bneg");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
      }
      break;
    case AND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "and");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case OR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        printf("%-5s", "or");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%s%d", "R", r3);
      }
      break;
    case NOT:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        printf("%-5s", "not");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
      }
      break;
    case EQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "eq");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case NEQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "neq");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case LE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "le");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case LT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "lt");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case GE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "ge");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case GT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "gt");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case MOVE:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        printf("%-5s", "move");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%s%d", "R", r2);
      }
      break;
    case LW:
      if (r1 >= 0 && r1 <= 5 && (memi % 4) == 0) {
        printf("%-5s", "lw");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case SW:
      if (r1 >= 0 && r1 <= 6 && (memi % 4) == 0) {
        printf("%-5s", "sw");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%05d", memi);
      }
      break;
    case LI:
      if (r1 >= 0 && r1 <= 5) {
        printf("%-5s", "li");
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
        printf("%2c", ' ');
        printf("%5d", memi);
      }
      break;
    case SYS:
      if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext <= 3) {
        printf("%-5s", "sys");
        printf("%2c", ' ');
        printf("%2d", ext);
        printf("%2c", ' ');
        printf("%s%d", "R", r1);
      } else if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext == 4) {
        printf("%-5s", "sys");
        printf("%2c", ' ');
        printf("%2d", ext);
      }

      break;

    default:

      return 0;
    }

    return 1;

  }

  else
    return 0;
}

/*
 * This function converts instructions into assembly language.
 * It prints the memory address of the instruction in hex followed by
 * the content of the instruction if it is valid or nothing if it is not.
 * Returns 1 if instruction is valid or 0 otherwise.
 *
 */
unsigned short disassemble(const Wrd program[], unsigned short num_instrs,
                           unsigned short *const bad_instr) {
  int i, bdins;

  if (program != NULL && bad_instr != NULL && (num_instrs <= NUM_WORDS)) {

    int mem = 0;
    for (i = 0; i < num_instrs; i++) {

      bdins = check_instruction(program[i]);

      if (bdins != 0) {
        printf("%04x", mem);
        printf(": ");
        print_instruction(program[i]);
        mem += 4;
        printf("\n");
      } else {
        *bad_instr = bdins;
        return 0;
      }
    }
    /* return 1; */
  }
  return 0;
}
/*
 * This function checks if a register is set before its given a value.
 * Returns -1 if instruction is valid or the index of the instruction
 * that modified the register.
 *
 */
short where_set(const Wrd program[], unsigned short num_words,
                unsigned short reg_nbr) {
  int i;

  if (program != NULL && (num_words <= NUM_WORDS) && reg_nbr >= 0 &&
      reg_nbr <= 5) {

    for (i = 0; i < num_words; i++) {
      if (check_instruction(program[i]) == 1) {

        int opc = (((1 << 5) - 1) & (program[i] >> (27)));
        int r1 = (((1 << 3) - 1) & (program[i] >> (21)));

        if (opc != 16 && opc != 17 && opc != 18 && opc != 19 && opc != 20 &&
            opc != 21 && opc != 24 && opc != 26 && reg_nbr == r1) {

          return i;

        } else if (opc == 26) {

          int ex = (((1 << 3) - 1) & (program[i] >> (24)));

          if ((ex == 0 || ex == 2) && reg_nbr == r1) {

            return i;
          }
        }
      }
    }
    return -1;

  }

  else
    return -1;
}
/*
 * This function checks if a register is set before its given a value.
 * Returns 1 if instructions in program are valid or sets pointer to first
 * unsafe instruction and returns a zero.
 *
 */
unsigned short is_safe(const Wrd program[], unsigned short pgm_size,
                       unsigned short *const bad_instr) {

  int i;

  if (program != NULL && (pgm_size <= NUM_WORDS) && bad_instr != NULL) {

    for (i = 0; i < pgm_size; i++) {

      Wrd op, r1, r2, r3, ext;
      Op_code ocode = op;
      op = (((1 << 5) - 1) & (program[i] >> (27)));
      ext = (((1 << 3) - 1) & (program[i] >> (24)));
      r1 = (((1 << 3) - 1) & (program[i] >> (21)));
      r2 = (((1 << 3) - 1) & (program[i] >> (18)));
      r3 = (((1 << 3) - 1) & (program[i] >> (15)));

      if (op >= 0 || op < 27) {
        switch (ocode) {
        case PLUS:
        case MINUS:
        case TIMES:
        case DIV:
        case MOD:
        case SHL:
        case SHR:
        case BAND:
        case BOR:
        case BXOR:
        case AND:
        case OR:

          if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
            if (where_set(program, i, r2) == -1 ||
                where_set(program, i, r3) == -1) {
              *bad_instr = i;

              return -1;
            }
          }

          break;

        case NEG:
        case ABS:
        case BNEG:
        case NOT:
          if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
            if (where_set(program, i, r2) == -1) {
              *bad_instr = i;

              return -1;
            }
          }
          break;

        case EQ:
        case NEQ:
        case LE:
        case LT:
        case GE:
        case GT:

          if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6) {
            if (where_set(program, i, r1) == -1 ||
                where_set(program, i, r2) == -1) {
              *bad_instr = i;

              return -1;
            }
          }
          break;

        case MOVE:
          if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
            if (where_set(program, i, r2) == -1) {
              *bad_instr = i;

              return -1;
            }
          }
          break;
        case LW:

          break;
        case SW:
          if (r1 >= 0 && r1 <= 6) {
            if (where_set(program, i, r1) == -1) {
              *bad_instr = i;

              return -1;
            }
          }
          break;
        case LI:

          break;
        case SYS:
          if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext <= 4) {
            if ((ext != 0) && (ext != 2) && (where_set(program, i, r1) == -1)) {
              *bad_instr = i;

              return -1;
            }
          }
          break;
        }
      }
    }

    return 1;
  }
  return 1;
}
/*
 * Helper function to check the validity of an instruction
 */
unsigned short check_instruction(Wrd instruction) {

  Wrd op, r1, r2, r3, memi, ext;
  Op_code ocode;
  op = (((1 << 5) - 1) & (instruction >> (27)));
  ext = (((1 << 3) - 1) & (instruction >> (24)));
  r1 = (((1 << 3) - 1) & (instruction >> (21)));
  r2 = (((1 << 3) - 1) & (instruction >> (18)));
  r3 = (((1 << 3) - 1) & (instruction >> (15)));
  memi = (((1 << 15) - 1) & (instruction >> (0)));

  ocode = op;
  if (op >= 0 || op < 27) {
    switch (ocode) {
    case PLUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }

      break;
    case MINUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case TIMES:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case DIV:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case MOD:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case NEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        return 1;
      }
      break;
    case ABS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        return 1;
      }
      break;
    case SHL:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case SHR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case BAND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case BOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case BXOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case BNEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        return 1;
      }
      break;
    case AND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case OR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
        return 1;
      }
      break;
    case NOT:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        return 1;
      }
      break;
    case EQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case NEQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case LE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case LT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case GE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case GT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case MOVE:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
        return 1;
      }
      break;
    case LW:
      if (r1 >= 0 && r1 <= 5 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case SW:
      if (r1 >= 0 && r1 <= 6 && (memi % 4) == 0) {
        return 1;
      }
      break;
    case LI:
      if (r1 >= 0 && r1 <= 5) {
        return 1;
      }
      break;
    case SYS:
      if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext <= 3) {
        return 1;
      } else if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext == 4) {
        return 1;
      }
      break;

    default:

      return 0;
    }

    return 0;

  }

  else
    return 0;
}