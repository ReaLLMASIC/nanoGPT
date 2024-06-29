// Copyright (c) 2024, Saligane's Group at University of Michigan and Google Research
//
// Licensed under the Apache License, Version 2.0 (the "License");

// you may not use this file except in compliance with the License.

// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//SiLU function: silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid.
//SiLU paper: https://arxiv.org/pdf/1702.03118

`ifndef __SILU_V_
`define __SILU_V_

module silu #(
    parameter   I_EXP   = 8,
    parameter   I_MNT   = 23,
    parameter   I_DATA  = I_EXP + I_MNT + 1
)(
    input       [I_DATA-1:0]    idata,
    output reg  [I_DATA-1:0]    odata
);

    // this need to be changed on fp16
    localparam const_p4 = {1'b0, 8'd129, {I_MNT{1'b0}}};
    localparam const_m4 = {1'b1, 8'd129, {I_MNT{1'b0}}};
    localparam const_1  = {1'b0, 8'd127, {I_MNT{1'b0}}};

    wire    [I_DATA-1:0]    ydet_p4, ydet_m4;

    wire [I_DATA-1:0] abs_x_div4;
    wire [I_DATA-1:0] x_div2;
    wire [I_DATA-1:0] m_x_a;
    wire [I_DATA-1:0] m_x_a_2;
    wire [I_DATA-1:0] outA;
    wire [I_DATA-1:0] outB;

    always @(*) begin
        case({ydet_p4[I_DATA-1], idata[I_DATA-1], ydet_m4[I_DATA-1]})
            3'b000: begin       // 4 < x
                odata = idata;
            end
            3'b100: begin       // 0 < x < 4
                odata = outB;
            end
            3'b110: begin       // -4 < x < 0
                odata = outA;
            end
            3'b111: begin       // x < -4
                odata = 0;
            end
            default: begin      // default
                odata = 0;
            end
        endcase
    end

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fadd_p4 (         // x - 4
        .a(idata),
        .b(const_p4),
        .z(ydet_p4),
        .rnd(3'b000),
        .op(1'b1),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fadd_m4 (         // x - (-4)
        .a(idata),
        .b(const_m4),
        .z(ydet_m4),
        .rnd(3'b000),
        .op(1'b1),
        .status()
    );

    fdiv4 #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) fdiv4_absxd4 (    // abs(x)/4
        .if32({1'b0, idata[I_DATA-2:0]}),
        .of32(abs_x_div4)
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fadd_1mx (        // 1 - abs(x)/4
        .a(const_1),
        .b(abs_x_div4),
        .z(m_x_a),
        .rnd(3'b000),
        .op(1'b1),
        .status()
    );

    DW_fp_square #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fsq_1m_x_a_2 (    // (1 - abs(x)/4)^2
        .a(m_x_a),
        .rnd(3'b000),
        .z(m_x_a_2),
        .status()
    );

    fdiv2 #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) fdiv2_xdiv2 (     // x/2
        .if32(idata),
        .of32(x_div2)
    );

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fmul_A (          // out -4 < x < 0
        .a(x_div2),
        .b(m_x_a_2),
        .rnd(3'b000),
        .z(outA),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP),
        .ieee_compliance(0)
    ) fadd_B (          // out 0 < x < 4
        .a(idata),
        .b(outA),
        .z(outB),
        .rnd(3'b000),
        .op(1'b1),
        .status()
    );

endmodule

`endif

