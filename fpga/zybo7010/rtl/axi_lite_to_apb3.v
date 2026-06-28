// SPDX-License-Identifier: Apache-2.0
// AXI4-Lite slave to APB3 master bridge for VenusCore control registers.

module axi_lite_to_apb3 #(
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_DATA_WIDTH = 32,
    parameter APB_ADDR_WIDTH = 12
) (
    input  wire                         clk,
    input  wire                         resetn,

    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    input  wire [AXI_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [(AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    output reg  [1:0]                   s_axi_bresp,
    output reg                          s_axi_bvalid,
    input  wire                         s_axi_bready,

    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    output reg  [AXI_DATA_WIDTH-1:0]    s_axi_rdata,
    output reg  [1:0]                   s_axi_rresp,
    output reg                          s_axi_rvalid,
    input  wire                         s_axi_rready,

    output reg  [APB_ADDR_WIDTH-1:0]    apb_paddr,
    output reg  [0:0]                   apb_psel,
    output reg                          apb_penable,
    output reg                          apb_pwrite,
    output reg  [31:0]                  apb_pwdata,
    input  wire [31:0]                  apb_prdata,
    input  wire                         apb_pready
);

    localparam ST_IDLE   = 2'd0;
    localparam ST_SETUP  = 2'd1;
    localparam ST_ACCESS = 2'd2;

    reg [1:0] state;
    reg [AXI_ADDR_WIDTH-1:0] awaddr_reg;
    reg [AXI_DATA_WIDTH-1:0] wdata_reg;
    reg aw_seen;
    reg w_seen;

    wire idle_and_free = (state == ST_IDLE) && !s_axi_bvalid && !s_axi_rvalid;
    wire aw_hs = s_axi_awvalid && s_axi_awready;
    wire w_hs = s_axi_wvalid && s_axi_wready;
    wire ar_hs = s_axi_arvalid && s_axi_arready;
    wire have_aw_next = aw_seen || aw_hs;
    wire have_w_next = w_seen || w_hs;
    wire [AXI_ADDR_WIDTH-1:0] selected_awaddr = aw_hs ? s_axi_awaddr : awaddr_reg;
    wire [AXI_DATA_WIDTH-1:0] selected_wdata = w_hs ? s_axi_wdata : wdata_reg;

    assign s_axi_awready = resetn && idle_and_free && !aw_seen;
    assign s_axi_wready  = resetn && idle_and_free && !w_seen;
    assign s_axi_arready = resetn && idle_and_free && !aw_seen && !w_seen &&
                           !s_axi_awvalid && !s_axi_wvalid;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            awaddr_reg <= {AXI_ADDR_WIDTH{1'b0}};
            wdata_reg <= {AXI_DATA_WIDTH{1'b0}};
            aw_seen <= 1'b0;
            w_seen <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_bvalid <= 1'b0;
            s_axi_rdata <= {AXI_DATA_WIDTH{1'b0}};
            s_axi_rresp <= 2'b00;
            s_axi_rvalid <= 1'b0;
            apb_paddr <= {APB_ADDR_WIDTH{1'b0}};
            apb_psel <= 1'b0;
            apb_penable <= 1'b0;
            apb_pwrite <= 1'b0;
            apb_pwdata <= 32'b0;
        end else begin
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end

            case (state)
                ST_IDLE: begin
                    apb_psel <= 1'b0;
                    apb_penable <= 1'b0;

                    if (aw_hs) begin
                        awaddr_reg <= s_axi_awaddr;
                        aw_seen <= 1'b1;
                    end
                    if (w_hs) begin
                        wdata_reg <= s_axi_wdata;
                        w_seen <= 1'b1;
                    end

                    if (ar_hs) begin
                        apb_paddr <= s_axi_araddr[APB_ADDR_WIDTH-1:0];
                        apb_pwrite <= 1'b0;
                        apb_pwdata <= 32'b0;
                        apb_psel <= 1'b1;
                        apb_penable <= 1'b0;
                        state <= ST_SETUP;
                    end else if (have_aw_next && have_w_next) begin
                        apb_paddr <= selected_awaddr[APB_ADDR_WIDTH-1:0];
                        apb_pwrite <= 1'b1;
                        apb_pwdata <= selected_wdata;
                        apb_psel <= 1'b1;
                        apb_penable <= 1'b0;
                        aw_seen <= 1'b0;
                        w_seen <= 1'b0;
                        state <= ST_SETUP;
                    end
                end

                ST_SETUP: begin
                    apb_penable <= 1'b1;
                    state <= ST_ACCESS;
                end

                ST_ACCESS: begin
                    if (apb_pready) begin
                        apb_psel <= 1'b0;
                        apb_penable <= 1'b0;
                        state <= ST_IDLE;
                        if (apb_pwrite) begin
                            s_axi_bresp <= 2'b00;
                            s_axi_bvalid <= 1'b1;
                        end else begin
                            s_axi_rdata <= apb_prdata;
                            s_axi_rresp <= 2'b00;
                            s_axi_rvalid <= 1'b1;
                        end
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    wire unused_wstrb = |s_axi_wstrb;

endmodule
