`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/12/7 12:21:10
// Design Name: 
// Module Name: mytest
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


module mytest();

reg clk;
reg rst;

wire [31:0] debug_reg1;
wire [31:0] debug_reg2;
wire [31:0] debug_reg3;
wire [31:0] pc;
wire [31:0] inst_d;
wire [31:0] data1;
wire [31:0] data2;
wire [31:0] data3;
wire [31:0] data4;
wire [31:0] data5;
// wire [31:0] res_d;
// wire [31:0] imm_d;
// wire alu_src_d;
// wire [31:0] RD22_d;

TOP top1(
.clk(clk), 
.rst(rst),
//debug
.debug_reg1(debug_reg1),
.debug_reg2(debug_reg2),
.debug_reg3(debug_reg3),
.nowpc(pc),
.inst_d(inst_d),
.data1(data1),
.data2(data2),
.data3(data3),
.data4(data4),
.data5(data5)
);

initial begin
    // Load instructions
    $readmemh("C:/Users/vJanGo/Desktop/bitAI/Grade3Autumn/����/project_1/testcode/instructions.txt", top1.im1.imem);
    // Load register initial values
    $readmemh("C:/Users/vJanGo/Desktop/bitAI/Grade3Autumn/����/project_1/testcode/register.txt", top1.rf1.regFiles);
    // Load memory data initial values
    $readmemh("C:/Users/vJanGo/Desktop/bitAI/Grade3Autumn/����/project_1/testcode/data_memory.txt", top1.dm1.dmem);
    $display("value: %h",top1.im1.imem[0]);
    rst = 1;
    clk = 0;
    #30 rst = 0; // 30ns ʱ�� CPU ��ʼ����
end





//lw test
//initial begin
//    top1.im1.imem[0] = 32'h00402183;
//    top1.im1.imem[1] = 32'h00302423;
//    top1.dm1.dmem[1] = 32'h00000077;
    
//    rst = 1;
//    clk = 0;
//    #30 rst = 0; // 30ns ʱ�� CPU ��ʼ����
//end

always
    #2 clk = ~clk; // ÿ�� 20ns ʱ���ź� clk ��תһ��
endmodule


//endmodule
