`timescale 1ns / 1ps
`include "macro.vh"

module cu(
    input [6:0] opcode,
    input [6:0] func7,
    input [2:0] func3,

    //��ת��ָ֧���ѡ���ź�,0��ʾ����ת,1��ʾjal,2��ʾbeq,3��ʾblt
    output reg [`BRANCH_LEN-1:0] branch,   
    // ALU�ڶ�����������ѡ���ź�,0��ʾ�ڶ����Ĵ�����1��ʾ������
    output reg alu_src,         
    // �Ĵ���д�����ݵ�ѡ���ź�, 0��ʾalu�����1��ʾ���ݴ洢����2��ʾ pc+4
    output reg [1:0] reg_src,   
    // �������ѡ���ź�,0Ϊ+��1Ϊ-,2Ϊ|
    output reg [`ALU_LEN-1:0] alu_op,  
    // �洢��дʹ��
    output reg dmem_we,        
    // �Ĵ�����дʹ��
    output reg reg_we           
    );

// ����ָ�����룬����������ź�
always@(*) begin
    case(opcode) 
       `OPCODE_R: begin //add ,sub
            if(func7 == `FUNC7_ADD) alu_op = 0;
            else if(func7 == `FUNC7_SUB) alu_op = 1;
            else if(func7 == `FUNC7_MUL) alu_op = 3;
            branch = 0;
            reg_src = 0;
            alu_src = 0;
            reg_we = 1;
            dmem_we = 0;
        end
        `OPCODE_I1: begin //addi,ori
            if (func3==`FUNC3_ADDI) alu_op = 0;
            else if(func3==`FUNC3_ORI) alu_op=2;
            branch=0;
            reg_src=0;
            alu_src = 1;
            reg_we = 1;
            dmem_we = 0;
       end
       `OPCODE_I2: begin// lw
            alu_op = 0;
            branch=0;
            reg_src = 1;
            alu_src = 1;
            reg_we = 1;
            dmem_we = 0;
        end
       `OPCODE_S: begin// sw
            alu_op = 0;
            branch=0;
            reg_src = 0;
            alu_src = 1;
            reg_we = 0;
            dmem_we = 1;
        end
       `OPCODE_B: begin
           if(func3 == `FUNC3_BEQ) begin// beq
               alu_op = 1;
               branch=2;
               reg_src = 2;
               alu_src = 0;
               reg_we = 0;
               dmem_we = 0;
            end
            else if(func3 == `FUNC3_BLT) begin //blt
               alu_op = 1;
               branch=3;
               reg_src = 2;
               alu_src = 0;
               reg_we = 0;
               dmem_we = 0;
            end
       end
       7'b1101111: begin//jal
           alu_op = 0;
           branch=1;
           reg_src = 2;
           alu_src = 0;
           reg_we = 1;
           dmem_we = 0;
       end
    endcase
end                
endmodule
