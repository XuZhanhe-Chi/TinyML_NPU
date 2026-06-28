// SPDX-License-Identifier: Apache-2.0
// ZYBO7010 wrapper: PS AXI4-Lite control + shared BRAM data path + VenusCoreTop.

module venuscore_zybo_wrapper #(
    parameter BRAM_ADDR_WIDTH = 17,
    parameter SHARED_MEM_BASE = 32'h4000_0000,
    parameter SHARED_MEM_BYTES = 32'h0002_0000
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF S_AXI:BRAM_PORT, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 50000000" *)
    input  wire                         s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aresetn, POLARITY ACTIVE_LOW" *)
    input  wire                         s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [31:0]                  s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                         s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire                         s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]                  s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]                   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                         s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire                         s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]                   s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire                         s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                         s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [31:0]                  s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire                         s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire                         s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0]                  s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]                   s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire                         s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 32, FREQ_HZ 50000000, HAS_BURST 0, HAS_LOCK 0, HAS_CACHE 0, HAS_PROT 1, HAS_QOS 0, HAS_REGION 0" *)
    input  wire                         s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT CLK" *)
    output wire                         bram_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT RST" *)
    output wire                         bram_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT EN" *)
    output wire                         bram_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT WE" *)
    output wire [3:0]                   bram_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT ADDR" *)
    output wire [BRAM_ADDR_WIDTH-1:0]   bram_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DIN" *)
    output wire [31:0]                  bram_wrdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DOUT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME BRAM_PORT, MEM_SIZE 131072, MEM_WIDTH 32, MASTER_TYPE BRAM_CTRL" *)
    input  wire [31:0]                  bram_rddata,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 venus_irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME venus_irq, SENSITIVITY LEVEL_HIGH" *)
    output wire                         venus_irq
);

    wire [11:0] apb_paddr;
    wire [0:0] apb_psel;
    wire apb_penable;
    wire apb_pready;
    wire apb_pwrite;
    wire [31:0] apb_pwdata;
    wire [31:0] apb_prdata;

    wire [31:0] ahb_haddr;
    wire ahb_hwrite;
    wire [2:0] ahb_hsize;
    wire [2:0] ahb_hburst;
    wire [3:0] ahb_hprot;
    wire [1:0] ahb_htrans;
    wire ahb_hmastlock;
    wire [31:0] ahb_hwdata;
    wire [31:0] ahb_hrdata;
    wire ahb_hready;
    wire ahb_hresp;

    axi_lite_to_apb3 u_ctrl_bridge (
        .clk(s_axi_aclk),
        .resetn(s_axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .apb_paddr(apb_paddr),
        .apb_psel(apb_psel),
        .apb_penable(apb_penable),
        .apb_pwrite(apb_pwrite),
        .apb_pwdata(apb_pwdata),
        .apb_prdata(apb_prdata),
        .apb_pready(apb_pready)
    );

    ahb_lite_to_bram_port #(
        .BASE_ADDR(SHARED_MEM_BASE),
        .MEM_BYTES(SHARED_MEM_BYTES),
        .BRAM_ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) u_data_bridge (
        .clk(s_axi_aclk),
        .resetn(s_axi_aresetn),
        .ahb_haddr(ahb_haddr),
        .ahb_hwrite(ahb_hwrite),
        .ahb_hsize(ahb_hsize),
        .ahb_hburst(ahb_hburst),
        .ahb_hprot(ahb_hprot),
        .ahb_htrans(ahb_htrans),
        .ahb_hmastlock(ahb_hmastlock),
        .ahb_hwdata(ahb_hwdata),
        .ahb_hrdata(ahb_hrdata),
        .ahb_hready(ahb_hready),
        .ahb_hresp(ahb_hresp),
        .bram_clk(bram_clk),
        .bram_rst(bram_rst),
        .bram_en(bram_en),
        .bram_we(bram_we),
        .bram_addr(bram_addr),
        .bram_wrdata(bram_wrdata),
        .bram_rddata(bram_rddata)
    );

    VenusCoreTop u_npu (
        .apb_s_PADDR(apb_paddr),
        .apb_s_PSEL(apb_psel),
        .apb_s_PENABLE(apb_penable),
        .apb_s_PREADY(apb_pready),
        .apb_s_PWRITE(apb_pwrite),
        .apb_s_PWDATA(apb_pwdata),
        .apb_s_PRDATA(apb_prdata),
        .ahb_m_HADDR(ahb_haddr),
        .ahb_m_HWRITE(ahb_hwrite),
        .ahb_m_HSIZE(ahb_hsize),
        .ahb_m_HBURST(ahb_hburst),
        .ahb_m_HPROT(ahb_hprot),
        .ahb_m_HTRANS(ahb_htrans),
        .ahb_m_HMASTLOCK(ahb_hmastlock),
        .ahb_m_HWDATA(ahb_hwdata),
        .ahb_m_HRDATA(ahb_hrdata),
        .ahb_m_HREADY(ahb_hready),
        .ahb_m_HRESP(ahb_hresp),
        .venus_irq(venus_irq),
        .clk(s_axi_aclk),
        .resetn(s_axi_aresetn)
    );

endmodule
