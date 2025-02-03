`timescale 1ns / 1ps
`include "macro.vh"

module alu(
    input [`ALU_LEN-1:0] op,
    input [31:0] in1,
    input [31:0] in2,
    output ZF,
    output SF,
    output reg [31:0] res
    );
    assign ZF=(res==0)?1:0;
    assign SF=res[31];

    always @(*) begin
        case (op)
            0: res=in1+in2;
            1: res=in1-in2;
            2: res=in1 | in2;
            3: res = in1*in2;
        endcase        
    end
endmodule
