import os
from simulator.network_simulator.app import Application, VideoApplicationBuilder
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_trace, Trace
from simulator.pantheon_dataset import PantheonDataset


def main():
     # /tank/patheon_traces/yihua/ethernet/queue20/2018-12-20T04-36-Colombia-to-AWS-Brazil-2-5-runs/quic_datalink_run1/formatted_bbr_datalink_run1.log
    trace = generate_trace(
        duration_range=(30, 30),
        bandwidth_lower_bound_range=(10, 10),
        bandwidth_upper_bound_range=(10, 10),
        delay_range=(50, 50),
        loss_rate_range=(0.0, 0.0),
        queue_size_range=(1, 1),
        T_s_range=(5, 5),
        delay_noise_range=(0, 0), seed=10)
    # # bbr_old = BBR_old(True)
    app = Application()
    # bbr = BBR(True, app=app)
    # bbr.test(trace, 'test', plot_flag=True)
    #
    # app = Application()
    # cubic = Cubic(True, app)
    # cubic.test(trace, 'test', plot_flag=True)

    # for loss in [0, 0.01]:
    #     for conn_type in ['ethernet', 'cellular']:
    #         dataset = PantheonDataset('../../../PCC-RL/data', conn_type, target_ccs=['bbr', 'vegas'])
    #         traces = dataset.get_traces(loss, 50)
    #         save_dirs = [os.path.join('test', str(loss), conn_type, link_name,
    #                                   trace_name) for link_name, trace_name in dataset.trace_names]
    #
    #         # cellular_dataset = PantheonDataset('../../data', 'cellular', target_ccs=['bbr', 'vegas'])
    #         # ethernet_dataset.get_traces(0, 100)
    #
    #         cubic.test_on_traces(traces, [os.path.join(save_dir, cubic.cc_name) for save_dir in save_dirs], True, 8)
    #         bbr_old.test_on_traces(traces, [os.path.join(save_dir, bbr_old.cc_name) for save_dir in save_dirs], True, 8)

    builder = VideoApplicationBuilder()
    # profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/test_mpeg.csv"
    profile = "/tank/pantheon_traces/test_mpeg.csv"
    tgt_br = 250 * 1000
    builder.set_profile(profile).set_fps(25).set_init_bitrate(tgt_br)
    app = builder.build()
    #bbr = BBR(True, app=app)

    # trace_file = "/tank/pantheon_traces/yihua/ethernet/queue20/2018-12-20T04-36-Colombia-to-AWS-Brazil-2-5-runs/quic_datalink_run1/bbr_datalink_run1.log"
    # trace = Trace.load_from_pantheon_file(trace_file, 0, 20)
    # trace.scale_bw(0.1, 5)
    #bbr.test(trace, 'test', plot_flag=True)
    #print(app.encoder.frame_id)
    #app = builder.build()
    cubic = Cubic(True, app)
    cubic.test(trace, 'test', plot_flag=True)
    print(app.get_summary())

if __name__ == "__main__":
    main()
