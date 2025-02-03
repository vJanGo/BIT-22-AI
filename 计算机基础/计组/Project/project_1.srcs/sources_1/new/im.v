`timescale 1ns / 1ps
`include "macro.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/12/6 22:02:35
// Design Name: 
// Module Name: im
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module im(
input [11:2] addr, // ָ��洢����32λ�Ĵ��������ʾ,ֱ��ȡ��������±ꣻ11��IM_SIZE����
    output [31:0] inst
    );

reg[31:0] imem[`IM_SIZE:0];
assign inst=imem[addr]; 
endmodule
