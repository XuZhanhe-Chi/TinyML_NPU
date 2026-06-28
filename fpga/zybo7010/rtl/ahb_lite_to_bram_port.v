// SPDX-License-Identifier: Apache-2.0
// Conservative AHB-Lite target bridge for routing VenusCore DMA to shared BRAM.

module ahb_lite_to_bram_port #(
    parameter BASE_ADDR = 32'h4000_0000,
    parameter MEM_BYTES = 32'h0002_0000,
    parameter BRAM_ADDR_WIDTH = 17,
    parameter READ_WAIT_CYCLES = 2
) (
    input  wire                         clk,
    input  wire                         resetn,

    input  wire [31:0]                  ahb_haddr,
    input  wire                         ahb_hwrite,
    input  wire [2:0]                   ahb_hsize,
    input  wire [2:0]                   ahb_hburst,
    input  wire [3:0]                   ahb_hprot,
    input  wire [1:0]                   ahb_htrans,
    input  wire                         ahb_hmastlock,
    input  wire [31:0]                  ahb_hwdata,
    output reg  [31:0]                  ahb_hrdata,
    output reg                          ahb_hready,
    output reg                          ahb_hresp,

    output wire                         bram_clk,
    output wire                         bram_rst,
    output reg                          bram_en,
    output reg  [3:0]                   bram_we,
    output reg  [BRAM_ADDR_WIDTH-1:0]   bram_addr,
    output reg  [31:0]                  bram_wrdata,
    input  wire [31:0]                  bram_rddata
);

    localparam ST_IDLE       = 2'd0;
    localparam ST_WRITE_DATA = 2'd1;
    localparam ST_READ_WAIT  = 2'd2;
    localparam ST_READ_RESP  = 2'd3;

    reg [1:0] state;
    reg [31:0] addr_reg;
    reg [2:0] size_reg;
    reg error_reg;
    reg [7:0] read_wait_cnt;

    assign bram_clk = clk;
    assign bram_rst = ~resetn;

    wire transfer_valid = ahb_hready && ahb_htrans[1];
    wire [31:0] local_addr = ahb_haddr - BASE_ADDR;
    wire in_range = (ahb_haddr >= BASE_ADDR) && (local_addr < MEM_BYTES);

    function [3:0] make_wstrb;
        input [2:0] size;
        input [1:0] addr_lsb;
        begin
            case (size)
                3'b000: make_wstrb = 4'b0001 << addr_lsb;
                3'b001: make_wstrb = addr_lsb[1] ? 4'b1100 : 4'b0011;
                default: make_wstrb = 4'b1111;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            addr_reg <= 32'b0;
            size_reg <= 3'b010;
            error_reg <= 1'b0;
            read_wait_cnt <= 8'b0;
            ahb_hrdata <= 32'b0;
            ahb_hready <= 1'b1;
            ahb_hresp <= 1'b0;
            bram_en <= 1'b0;
            bram_we <= 4'b0000;
            bram_addr <= {BRAM_ADDR_WIDTH{1'b0}};
            bram_wrdata <= 32'b0;
        end else begin
            bram_en <= 1'b0;
            bram_we <= 4'b0000;
            ahb_hresp <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ahb_hready <= 1'b1;
                    if (transfer_valid) begin
                        addr_reg <= local_addr;
                        size_reg <= ahb_hsize;
                        error_reg <= !in_range;
                        ahb_hready <= 1'b0;
                        if (ahb_hwrite) begin
                            state <= ST_WRITE_DATA;
                        end else begin
                            bram_en <= in_range;
                            bram_addr <= local_addr[BRAM_ADDR_WIDTH-1:0];
                            read_wait_cnt <= (READ_WAIT_CYCLES <= 0) ? 8'd0 : (READ_WAIT_CYCLES - 1);
                            state <= ST_READ_WAIT;
                        end
                    end
                end

                ST_WRITE_DATA: begin
                    bram_en <= !error_reg;
                    bram_we <= error_reg ? 4'b0000 : make_wstrb(size_reg, addr_reg[1:0]);
                    bram_addr <= addr_reg[BRAM_ADDR_WIDTH-1:0];
                    bram_wrdata <= ahb_hwdata;
                    ahb_hready <= 1'b1;
                    ahb_hresp <= error_reg;
                    state <= ST_IDLE;
                end

                ST_READ_WAIT: begin
                    if (read_wait_cnt == 0) begin
                        state <= ST_READ_RESP;
                    end else begin
                        read_wait_cnt <= read_wait_cnt - 1'b1;
                    end
                end

                ST_READ_RESP: begin
                    ahb_hrdata <= error_reg ? 32'b0 : bram_rddata;
                    ahb_hready <= 1'b1;
                    ahb_hresp <= error_reg;
                    state <= ST_IDLE;
                end

                default: begin
                    ahb_hready <= 1'b1;
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    wire unused_sideband = |{ahb_hburst, ahb_hprot, ahb_hmastlock};

endmodule
