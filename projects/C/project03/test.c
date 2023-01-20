#include "machine.h"
#include <assert.h>
#include <stdio.h>

#define PROGRAM_SIZE 10
unsigned short check_instruction(Wrd instruction);

unsigned short is_safe(const Wrd program[], unsigned short pgm_size,
                       unsigned short *const bad_instr);
short where_set(const Wrd program[], unsigned short num_words,
                unsigned short reg_nbr);

unsigned short check_instruction(Wrd instruction) {

  Wrd op, r1, r2, r3, memi, ext;
  op = (((1 << 5) - 1) & (instruction >> (27)));
  ext = (((1 << 3) - 1) & (instruction >> (24)));
  r1 = (((1 << 3) - 1) & (instruction >> (21)));
  r2 = (((1 << 3) - 1) & (instruction >> (18)));
  r3 = (((1 << 3) - 1) & (instruction >> (15)));
  memi = (((1 << 15) - 1) & (instruction >> (0)));
  Op_code ocode = op;
  if (op >= 0 || op < 27) {
    switch (ocode) {
    case PLUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }

      break;
    case MINUS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case TIMES:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case DIV:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case MOD:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case NEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
      }
      break;
    case ABS:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
      }
      break;
    case SHL:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case SHR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case BAND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case BOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case BXOR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case BNEG:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
      }
      break;
    case AND:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case OR:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6 && r3 >= 0 && r3 <= 6) {
      }
      break;
    case NOT:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
      }
      break;
    case EQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case NEQ:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case LE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case LT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case GE:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case GT:
      if (r1 >= 0 && r1 <= 6 && r2 >= 0 && r2 <= 6 && (memi % 4) == 0) {
      }
      break;
    case MOVE:
      if (r1 >= 0 && r1 <= 5 && r2 >= 0 && r2 <= 6) {
      }
      break;
    case LW:
      if (r1 >= 0 && r1 <= 5 && (memi % 4) == 0) {
      }
      break;
    case SW:
      if (r1 >= 0 && r1 <= 6 && (memi % 4) == 0) {
      }
      break;
    case LI:
      if (r1 >= 0 && r1 <= 5 && (memi % 4) == 0) {
      }
      break;
    case SYS:
      if (r1 >= 0 && r1 <= 5 && ext >= 0 && ext <= 4) {
      }
      break;

    default:
      break;
    }

    return 1;
  } else
    return 0;
}

int bin(unsigned n) {
  unsigned i;
  for (i = 1 << 31; i > 0; i = i / 2)
    (n & i) ? printf("1") : printf("0");

  return 0;
}

int main(void) {
  int d;
  Wrd program[PROGRAM_SIZE] = {0xc8200014, 0xc8400006, 0x08650000, 0x688d0000,
                               0xc8200000, 0x68ac8000, 0x70ac8000, 0x78a40000,
                               0x78880000, 0x284c0000};

  unsigned short location = -1;

  d = (is_safe(program, PROGRAM_SIZE, &location) == 1);
  assert(is_safe(program, PROGRAM_SIZE, &location) == 1);
  printf("%d\n", d);

  return 0;

  /*
  unsigned inst = 3358064856;
  unsigned maskop = 4160749568;
  unsigned maskr1 = 14680064;
  unsigned maskr2 = 1835008;
  unsigned maskr3 = 229376;



  unsigned int op,r1,r2,r3,memi,ext;
  op = (((1<<5)-1)&(inst>>(27)));
  ext = (((1<<3)-1)&(inst>>(24)));
  r1 = (((1<<3)-1)&(inst>>(21)));
  r2 = (((1<<3)-1)&(inst>>(18)));
  r3 = (((1<<3)-1)&(inst>>(15)));
  memi = (((1<<15)-1)&(inst));
  Op_code ocode = op;

  bin(inst);
  printf("\n");



  Wrd arr[5] = {0};
  int * pt = arr;

  if (pt == NULL)
  {
      printf("null");
  }


 int mem = 0;
 int i;
 for ( i = 0; i < 10; i++)
 {

     printf("%04x",mem);
     printf(": \n");
     mem += 4;
 }

 if (check_instruction(3358064856) != 0)
 {
     printf("not 0\n");
     printf("%d\n",check_instruction(3358064856));


 }
 printf("%d",((-10) && 1));

 */
  /*
      bin(op);
      printf("\n");
      bin(r1);
      printf("\n");
      bin(r2);
      printf("\n");
      bin(r3);
      printf("\n");


      printf("opcode : %d\n",op);
      printf("R1 : %d\n",r1);
      printf("R2 : %d\n",r2);

      printf("R3 : %d\n",r3);



       printf("opcode : %d\n",ocode);

      */
}

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

unsigned short is_safe(const Wrd program[], unsigned short pgm_size,
                       unsigned short *const bad_instr) {

  int i;

  if (program != NULL && (pgm_size <= NUM_WORDS) && bad_instr != NULL) {

    for (i = 0; i < pgm_size; i++) {

      Wrd op, r1, r2, r3, memi, ext;
      op = (((1 << 5) - 1) & (program[i] >> (27)));
      ext = (((1 << 3) - 1) & (program[i] >> (24)));
      r1 = (((1 << 3) - 1) & (program[i] >> (21)));
      r2 = (((1 << 3) - 1) & (program[i] >> (18)));
      r3 = (((1 << 3) - 1) & (program[i] >> (15)));
      memi = (((1 << 15) - 1) & (program[i] >> (0)));
      Op_code ocode = op;

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

        default:
          break;
        }
      }
    }

    return 1;
  }
  return -1;
}