// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps

module tb_axi_lite_to_apb3;
  reg clk = 1'b0;
  reg resetn = 1'b0;
  always #5 clk = ~clk;

  reg [31:0] awaddr = 0;
  reg awvalid = 0;
  wire awready;
  reg [31:0] wdata = 0;
  reg [3:0] wstrb = 4'hf;
  reg wvalid = 0;
  wire wready;
  wire [1:0] bresp;
  wire bvalid;
  reg bready = 0;
  reg [31:0] araddr = 0;
  reg arvalid = 0;
  wire arready;
  wire [31:0] rdata;
  wire [1:0] rresp;
  wire rvalid;
  reg rready = 0;

  wire [11:0] paddr;
  wire [0:0] psel;
  wire penable;
  wire pwrite;
  wire [31:0] pwdata;
  reg [31:0] prdata = 0;
  reg pready = 0;

  axi_lite_to_apb3 dut (
    .clk(clk), .resetn(resetn),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .apb_paddr(paddr), .apb_psel(psel), .apb_penable(penable),
    .apb_pwrite(pwrite), .apb_pwdata(pwdata), .apb_prdata(prdata), .apb_pready(pready)
  );

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

  task automatic send_aw;
    input [31:0] address;
    begin
      @(negedge clk);
      awaddr = address;
      awvalid = 1'b1;
      while (!awready) @(negedge clk);
      @(posedge clk);
      @(negedge clk);
      awvalid = 1'b0;
    end
  endtask

  task automatic send_w;
    input [31:0] data;
    begin
      @(negedge clk);
      wdata = data;
      wstrb = 4'hf;
      wvalid = 1'b1;
      while (!wready) @(negedge clk);
      @(posedge clk);
      @(negedge clk);
      wvalid = 1'b0;
    end
  endtask

  task automatic send_ar;
    input [31:0] address;
    begin
      @(negedge clk);
      araddr = address;
      arvalid = 1'b1;
      while (!arready) @(negedge clk);
      @(posedge clk);
      @(negedge clk);
      arvalid = 1'b0;
    end
  endtask

  task automatic complete_apb_write;
    input [11:0] expected_addr;
    input [31:0] expected_data;
    begin
      check(psel && !penable && pwrite, "APB write setup phase missing");
      check(paddr == expected_addr, "APB write address mismatch");
      check(pwdata == expected_data, "APB write data mismatch");
      @(posedge clk); #1;
      check(psel && penable && pwrite, "APB write access phase missing");
      repeat (2) begin
        @(posedge clk); #1;
        check(psel && penable && !bvalid, "APB write did not hold during wait state");
      end
      @(negedge clk); pready = 1'b1;
      @(posedge clk); #1;
      pready = 1'b0;
      check(bvalid && bresp == 2'b00, "AXI write response missing");
      repeat (2) begin
        @(posedge clk); #1;
        check(bvalid, "AXI BVALID did not survive backpressure");
      end
      @(negedge clk); bready = 1'b1;
      @(posedge clk); #1;
      @(negedge clk); bready = 1'b0;
      check(!bvalid, "AXI write response did not retire");
    end
  endtask

  initial begin
    repeat (3) @(posedge clk);
    resetn = 1'b1;
    @(posedge clk); #1;
    check(awready && wready && arready, "AXI channels not ready after reset");

    // AW may arrive before W.
    send_aw(32'h43c0_0010);
    repeat (2) @(posedge clk);
    check(!awready && wready && !psel, "AW-only state was not retained");
    send_w(32'h1122_3344);
    complete_apb_write(12'h010, 32'h1122_3344);

    // W may arrive before AW.
    send_w(32'ha5a5_5a5a);
    repeat (2) @(posedge clk);
    check(awready && !wready && !psel, "W-only state was not retained");
    send_aw(32'h43c0_0004);
    complete_apb_write(12'h004, 32'ha5a5_5a5a);

    // Read response must preserve data while RREADY is low.
    prdata = 32'h0005_0000;
    send_ar(32'h43c0_000c);
    check(psel && !penable && !pwrite, "APB read setup phase missing");
    @(posedge clk); #1;
    check(psel && penable && !pwrite, "APB read access phase missing");
    repeat (2) begin
      @(posedge clk); #1;
      check(psel && penable && !rvalid, "APB read did not hold during wait state");
    end
    @(negedge clk); pready = 1'b1;
    @(posedge clk); #1;
    pready = 1'b0;
    check(rvalid && rresp == 2'b00 && rdata == 32'h0005_0000,
          "AXI read response mismatch");
    repeat (2) begin
      @(posedge clk); #1;
      check(rvalid && rdata == 32'h0005_0000, "AXI RVALID/data did not survive backpressure");
    end
    @(negedge clk); rready = 1'b1;
    @(posedge clk); #1;
    @(negedge clk); rready = 1'b0;
    check(!rvalid, "AXI read response did not retire");

    // Reset must discard a partially collected write.
    send_aw(32'h43c0_0080);
    @(negedge clk); resetn = 1'b0;
    repeat (2) @(posedge clk);
    @(negedge clk); resetn = 1'b1;
    @(posedge clk); #1;
    check(!psel && !penable && !bvalid && !rvalid, "reset left stale bridge state");
    check(awready && wready && arready, "bridge did not recover after reset");

    $display("[PASS] axi_lite_to_apb3 protocol regression");
    $finish;
  end

  initial begin
    #20000;
    $display("[FAIL] timeout");
    $fatal(1);
  end
endmodule
