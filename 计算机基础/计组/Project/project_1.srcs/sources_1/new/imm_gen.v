`timescale 1ns / 1ps
`include "macro.vh"

module imm_gen(
    input [31:0] inst,
    output reg [31:0] imm
);

    // ����ָ�����������
    always @(*) begin
        case(inst[6:0])
            `OPCODE_I1, `OPCODE_I2:  // addi, lw
                imm = { {20{inst[31]}}, inst[31:20] };  // ������չ12λ������

            `OPCODE_S:  // sw
                imm = { {20{inst[31]}}, inst[31:25], inst[11:7] };  // ������չ12λ������

            `OPCODE_B:  // beq, blt
                begin
                    imm[31:13] = {19{inst[31]}};
                    imm[12:0] = {inst[31], inst[7], inst[30:25], inst[11:8], 1'b0};
                end

            `OPCODE_J:  // jal
                imm = { {11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0 };  // ������չ21λ������
        endcase
    end

endmodule