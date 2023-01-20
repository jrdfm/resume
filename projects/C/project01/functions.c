#include "functions.h"



int prism_area(int l, int w, int h) {
    int a;
  if(l > 0 && w > 0 && h > 0 )
    {
      a = 2*(w*l+l*h+w*h);
      return a;
    }
  else return -1;

}

long jacobsthal(short n) {
    if(n<0)
    {
     return -1;
    }
  if(n == 0)
    {
     return 0;
    } 
  if(n == 1)
    {
      return 1;
    }
   
  return jacobsthal(n-1)+2*jacobsthal(n-2);

}

short ith_digit(long n, unsigned short i) {
   short r;
   int j,p;

  if(i == 0)
    {
      return -1;
    }
    p = 1;

  for(j = 0;j< i-1;++j)
    {
      p *= 10;
    }
   r = n/p;
   r = r % 10;
  
  if(r == 0)
    {
      return -1;
    }
  if(r < 0)
    {
      r =-r;
    }
  return r;

}
