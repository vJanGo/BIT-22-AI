`timescale 1ns / 1ps
`include "macro.vh"

module pc(
input  clk, 
    input  rst, 
    input [31:0] npc, 
    output reg [31:0] pc  
    );

    always @(posedge clk, posedge rst) begin
        if (rst) 
            pc <= `DEFAULT_VAL; 
        else 
            pc <= npc;  
    end
endmodule
