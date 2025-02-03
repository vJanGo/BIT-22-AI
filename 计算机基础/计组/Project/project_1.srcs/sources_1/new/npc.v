`timescale 1ns / 1ps

module npc(
    input [31:0] imm,    
    input jump,          
    input [31:0] pc,     
    output reg [31:0] npc,   
    output [31:0] pc_4   
);

    
    assign pc_4 = pc + 4;

    
    always @(*) begin
        if (jump == 0)
            npc = pc_4;
        else
            npc = pc + imm;
    end

endmodule

