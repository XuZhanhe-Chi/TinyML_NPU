// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps

module tb_ahb_lite_to_bram_port;
  localparam [31:0] BASE = 32'h4000_0000;

  reg clk = 1'b0;
  reg resetn = 1'b0;
  always #5 clk = ~clk;

  reg [31:0] haddr = 0;
  reg hwrite = 0;
  reg [2:0] hsize = 3'b010;
  reg [2:0] hburst = 0;
  reg [3:0] hprot = 0;
  reg [1:0] htrans = 0;
  reg hmastlock = 0;
  reg [31:0] hwdata = 0;
  wire [31:0] hrdata;
  wire hready;
  wire hresp;
  wire bram_clk;
  wire bram_rst;
  wire bram_en;
  wire [3:0] bram_we;
  wire [16:0] bram_addr;
  wire [31:0] bram_wrdata;
  reg [31:0] bram_rddata = 0;
  reg [31:0] memory [0:63];
  integer i;

  ahb_lite_to_bram_port #(
    .BASE_ADDR(BASE), .MEM_BYTES(32'h0000_0100),
    .BRAM_ADDR_WIDTH(17), .READ_WAIT_CYCLES(2)
  ) dut (
    .clk(clk), .resetn(resetn), .ahb_haddr(haddr), .ahb_hwrite(hwrite),
    .ahb_hsize(hsize), .ahb_hburst(hburst), .ahb_hprot(hprot),
    .ahb_htrans(htrans), .ahb_hmastlock(hmastlock), .ahb_hwdata(hwdata),
    .ahb_hrdata(hrdata), .ahb_hready(hready), .ahb_hresp(hresp),
    .bram_clk(bram_clk), .bram_rst(bram_rst), .bram_en(bram_en),
    .bram_we(bram_we), .bram_addr(bram_addr), .bram_wrdata(bram_wrdata),
    .bram_rddata(bram_rddata)
  );

  always @(posedge clk) begin
    if (bram_en) begin
      bram_rddata <= memory[bram_addr[7:2]];
      if (bram_we[0]) memory[bram_addr[7:2]][7:0] <= bram_wrdata[7:0];
      if (bram_we[1]) memory[bram_addr[7:2]][15:8] <= bram_wrdata[15:8];
      if (bram_we[2]) memory[bram_addr[7:2]][23:16] <= bram_wrdata[23:16];
      if (bram_we[3]) memory[bram_addr[7:2]][31:24] <= bram_wrdata[31:24];
    end
  end

  task automatic check;
    input condition;
    input [8*96-1:0] message;
    begin
      if (!condition) begin
        $display("[FAIL] %0s", message);
        $fatal(1);
      end
    end
  endtask

  task automatic ahb_write;
    input [31:0] address;
    input [2:0] size;
    input [31:0] data;
    input [3:0] expected_strobe;
    input expected_error;
    begin
      while (!hready) @(negedge clk);
      @(negedge clk);
      haddr = address;
      hwrite = 1'b1;
      hsize = size;
      htrans = 2'b10;
      @(posedge clk); #1;
      check(!hready, "write address phase did not insert wait");
      @(negedge clk);
      htrans = 2'b00;
      hwdata = data;
      @(posedge clk); #1;
      check(hready, "write response did not complete");
      check(hresp == expected_error, "write HRESP mismatch");
      check(bram_we == (expected_error ? 4'b0000 : expected_strobe), "write strobe mismatch");
      if (!expected_error) check(bram_addr == address - BASE, "write BRAM address mismatch");
      @(posedge clk); #1;
    end
  endtask

  task automatic ahb_read;
    input [31:0] address;
    input [31:0] expected_data;
    input expected_error;
    begin
      while (!hready) @(negedge clk);
      @(negedge clk);
      haddr = address;
      hwrite = 1'b0;
      hsize = 3'b010;
      htrans = 2'b10;
      @(posedge clk); #1;
      check(!hready, "read address phase did not insert wait");
      @(negedge clk); htrans = 2'b00;
      while (!hready) begin
        @(posedge clk); #1;
      end
      check(hresp == expected_error, "read HRESP mismatch");
      check(hrdata == (expected_error ? 32'b0 : expected_data), "read data mismatch");
      @(posedge clk); #1;
    end
  endtask

  initial begin
    for (i = 0; i < 64; i = i + 1) memory[i] = 32'b0;
    repeat (3) @(posedge clk);
    resetn = 1'b1;
    @(posedge clk); #1;
    check(hready && !hresp && !bram_en, "bridge reset state mismatch");
    check(bram_clk === clk && !bram_rst, "BRAM clock/reset mapping mismatch");

    ahb_write(BASE + 0, 3'b010, 32'haabb_ccdd, 4'b1111, 1'b0);
    check(memory[0] == 32'haabb_ccdd, "word write failed");
    ahb_write(BASE + 1, 3'b000, 32'h0000_5500, 4'b0010, 1'b0);
    check(memory[0] == 32'haabb_55dd, "byte write failed");
    ahb_write(BASE + 2, 3'b001, 32'h6677_0000, 4'b1100, 1'b0);
    check(memory[0] == 32'h6677_55dd, "halfword write failed");

    ahb_read(BASE + 0, 32'h6677_55dd, 1'b0);

    // Consecutive transfers start as soon as HREADY returns.
    ahb_write(BASE + 4, 3'b010, 32'h1111_2222, 4'b1111, 1'b0);
    ahb_write(BASE + 8, 3'b010, 32'h3333_4444, 4'b1111, 1'b0);
    check(memory[1] == 32'h1111_2222 && memory[2] == 32'h3333_4444,
          "consecutive writes failed");

    ahb_write(BASE + 32'h100, 3'b010, 32'hdead_beef, 4'b0000, 1'b1);
    ahb_read(BASE + 32'h100, 32'b0, 1'b1);

    // Reset while a read is waiting must restore an idle response.
    @(negedge clk);
    haddr = BASE + 4;
    hwrite = 1'b0;
    htrans = 2'b10;
    @(posedge clk); #1;
    check(!hready, "read was not accepted before reset test");
    @(negedge clk);
    htrans = 2'b00;
    resetn = 1'b0;
    repeat (2) @(posedge clk);
    @(negedge clk); resetn = 1'b1;
    @(posedge clk); #1;
    check(hready && !hresp && !bram_en && bram_we == 0, "reset did not clear read state");

    $display("[PASS] ahb_lite_to_bram_port protocol regression");
    $finish;
  end

  initial begin
    #20000;
    $display("[FAIL] timeout");
    $fatal(1);
  end
endmodule
