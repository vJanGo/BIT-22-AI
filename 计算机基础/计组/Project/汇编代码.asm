.data
.word 20, 5, 4, 3, 2, 1  # 20 为数组字节数
.text
addi a0, x0, 4        # a0 = 数组元素首地址，跳过字节数
lw a1, 0(x0)        # a1 = 数组字节数（20）
sub t0, t0, t0        # 清零 t0，外层循环计数器归零
addi t1, x0, 4        # t1 = 4，初始化内层循环计数器，指向数组第二个元素
loop1_start:
    sub t0, t0, t0    # t0 = 0，外层循环计数器归零
    addi t1, x0, 4    # t1 = 4，重新初始化内层循环计数器
loop2_start:
    blt t1, a1, loop2_body   # 如果 t1 < a1 则跳转到 loop2_body
    jal x1, loop1_end         # 否则结束内层循环，跳转到外层循环结束

loop2_body:
    add t2, a0, t1            # t2 = a0 + t1，t2 是当前比较的元素地址
    lw t3, -4(t2)             # 取出当前元素和前一个元素
    lw t4, 0(t2)
    blt t3, t4, loop2_end     # 如果前一个元素 t3 小于当前元素 t4，跳过交换
    sw t3, 0(t2)              # 否则交换
    sw t4, -4(t2)
    addi t0, t0, 1            # 外层计数器自增

loop2_end:
    addi t1, t1, 4            # 更新内层循环计数器
    jal x1, loop2_start       # 返回到内层循环开始

loop1_end:
    beq t0, x0, stop          # 如果 t0 == 0（没有发生交换），则排序结束
    jal x1, loop1_start       # 否则，回到外层循环继续
stop:

