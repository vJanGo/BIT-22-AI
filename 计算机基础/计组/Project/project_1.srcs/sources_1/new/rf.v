`timescale 1ns / 1ps
`include "macro.vh"

module rf(
    input clk,
    input rst,
    input we,           // дʹ��
    input [4:0] RA1,    // ����ַ1 (�Ĵ���������±�)
    input [4:0] RA2,    // ����ַ2
    input [4:0] WA,     // д��ַ
    input [31:0] WD,    // д����
    output [31:0] RD1,  // ����������1
    output [31:0] RD2,  // ����������2
    output [31:0] debug_reg1,
    output [31:0] debug_reg2,
    output [31:0] debug_reg3
);

    reg [31:0] regFiles [0:31];
    wire [31:0] writeData;
    wire writeEnable;
    wire [4:0] writeAddress;
    wire [31:0] defaultVal;

    assign writeData = WD;
    assign writeEnable = we && !rst && (WA != 0);
    assign writeAddress = WA;
    assign defaultVal = `DEFAULT_VAL;

    // д�����߼�
    always @(posedge clk) begin
        if (writeEnable) begin
            regFiles[writeAddress] <= writeData;
        end
    end

    // �������߼�
    wire [31:0] readData1;
    wire [31:0] readData2;

    assign readData1 = (RA1 == 0) ? defaultVal : regFiles[RA1];
    assign readData2 = (RA2 == 0) ? defaultVal : regFiles[RA2];

    assign RD1 = readData1;
    assign RD2 = readData2;


    // �������
    assign debug_reg1 = regFiles[1];
    assign debug_reg2 = regFiles[2];
    assign debug_reg3 = regFiles[3];

endmodule