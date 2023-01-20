# -*- mode: text -*-

#
#

     .data 

number: .word 0
result: .word 0


     .text

main: li    $sp, 0x7ffffffc  # set up stack pointer
      li    $v0,5 # read in number
      syscall
      move  $t0,$v0
      sw    $t0,number
      sw    $t0,($sp) # push number onto stack
      sub   $sp,$sp,4

      jal   jacobsthal  # call jacobsthal

      add   $sp,$sp,4  # pop number 
      move  $t0,$v0    # store return to result
      sw    $t0,result 

      move  $a0, $t0 # print result
      li    $v0,1 
      syscall
      
      li    $v0,11 # print '\n'
      li    $a0,10
      syscall

      li    $v0,10
      syscall

                                     # prologue
helper:     sub     $sp, $sp, 8      # set new stack pointer
            sw      $ra, 8($sp)      # save return address in stack
            sw      $fp, 4($sp)      # save old frame pointer in stack
            add     $fp, $sp, 8      # set new frame pointer

            lw      $t0, 8($fp)      # get x in caller's frame
            lw      $t1, 4($fp)      # get y in caller's frame
            mul     $t0,$t0,2        # $t0 = 2 * x
            add     $t1,$t0,$t1      # $t1 = 2 * x + y 

            move    $v0, $t1         # return  2 * x + y 



                                     # epilogue
            lw      $ra, 8($sp)      # load return address from stack
            lw      $fp, 4($sp)      # restore old frame pointer from stack
            add     $sp, $sp, 8      # reset stack pointer
            jr      $ra              # return to caller using saved return address


                                     # prologue
jacobsthal:     sub     $sp, $sp, 24      # set new stack pointer
                sw      $ra, 24($sp)      # save return address in stack
                sw      $fp, 20($sp)      # save old frame pointer in stack
                add     $fp, $sp, 24      # set new frame pointer

                li      $t0,-1
                sw      $t0,16($sp)    # ans = -1

                lw      $t2,4($fp)    # get n in caller's frame

loop:       bltz     $t2,endif  # if(n < 0 ) end

            seq     $t3,$t2,0 # $t5 = (n == 0)
            seq     $t4,$t2,1 # $t6 = (n == 1)
            or      $t3,$t3,$t4 # t5 = (n == 0 || n == 1)

            beqz    $t3,else 

            
            sw      $t2,16($sp)    # ans = n

                    # return  ans 
            j       endif

else:       li      $t3,1
            sw      $t3,16($sp)   # ans = 1
            li      $t4,0
            sw      $t4,12($sp)   # prev = 0

            li      $t5,2
            sw      $t5,4($sp) # i = 2

            j       for

for:        bgt     $t5,$t2,endfor  # if(i > n) endfor

            lw      $t4,12($sp) # get prev from stack
            lw      $t3,16($sp) # get ans from stack

        
            sw    $t4,($sp) # push prev onto stack
            sub   $sp,$sp,4
            sw    $t3,($sp) # push ans onto stack
            sub   $sp,$sp,4


            jal   helper  # call helper

            add   $sp,$sp,8 # pop arguments from stack

            move  $t1,$v0 # 
            sw    $t1,8($sp) # temp = helper(prev,ans)

            sw    $t3,12($sp) # prev = ans
            sw    $t1,16($sp) # ans = temp

            add   $t5,$t5,1
            sw    $t5,4($sp)  # i ++

            j     for
             
endfor:     j endif


endif:    
            lw      $t0,16($sp)
            move    $v0, $t0                 
                                     # epilogue
            lw      $ra, 24($sp)      # load return address from stack
            lw      $fp, 20($sp)      # restore old frame pointer from stack
            add     $sp, $sp, 24      # reset stack pointer
            jr      $ra              # return to caller using saved return address