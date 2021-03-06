import os
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_trace
from simulator.pantheon_dataset import PantheonDataset


def main():
    # trace = generate_trace(
    #     duration_range=(30, 30),
    #     bandwidth_lower_bound_range=(5, 5),
    #     bandwidth_upper_bound_range=(5, 5),
    #     delay_range=(50, 50),
    #     loss_rate_range=(0, 0),
    #     queue_size_range=(1, 1),
    #     T_s_range=(30, 30),
    #     delay_noise_range=(0, 0), seed=10)
    bbr_old = BBR_old(True)
    cubic = Cubic(True)
    # cubic.test(trace, 'test', plot_flag=True)

    for loss in [0, 0.01]:
        for conn_type in ['ethernet', 'cellular']:
            dataset = PantheonDataset('../../../PCC-RL/data', conn_type, target_ccs=['bbr', 'vegas'])
            traces = dataset.get_traces(loss, 50)
            save_dirs = [os.path.join('test', str(loss), conn_type, link_name,
                                      trace_name) for link_name, trace_name in dataset.trace_names]

            # cellular_dataset = PantheonDataset('../../data', 'cellular', target_ccs=['bbr', 'vegas'])
            # ethernet_dataset.get_traces(0, 100)

            cubic.test_on_traces(traces, [os.path.join(save_dir, cubic.cc_name) for save_dir in save_dirs], True, 8)
            bbr_old.test_on_traces(traces, [os.path.join(save_dir, bbr_old.cc_name) for save_dir in save_dirs], True, 8)


if __name__ == "__main__":
    main()
