#!/usr/bin/env nu

module rule {
    export def add [--device: string = "eno1", ...options, --dry] {
        let args = [add dev $device root netem ...$options]
        print $'sudo tc qdisc ($args | str join " ")'
        if not $dry {
            sudo tc qdisc ...$args
        }
    }

    export def change [--device: string = "eno1", ...options, --dry] {
        let args = [change dev $device root netem ...$options]
        print $'sudo tc qdisc ($args | str join " ")'
        if not $dry {
            sudo tc qdisc ...$args
        }
    }

    export def del [--device: string = "eno1", --dry] {
        let args = [del dev $device root netem]
        print $'sudo tc qdisc ($args | str join " ")'
        if not $dry {
            sudo tc qdisc ...$args
        }
    }
}

def run-rules [
    rules: table<time: duration, opts: list<string>>
    tail: duration
    --device: string = "eno1"
    --dry
] {
    use rule

    print $'Start, dry: ($dry)'
    sleep $tail

    let first = $rules | first
    rule add ...$first.opts --device=$device --dry=$dry
    sleep $first.time

    $rules | skip 1 | each {|rule|
        rule change ...$rule.opts --device=$device --dry=$dry
        sleep $rule.time
    }

    rule del --device=$device --dry=$dry
    sleep $tail

    print 'Currently defined rules:'
    tc -p qdisc ls dev $device
    print 'Done'
}

const STEPS_DEEP = [
    [time opts];
    [15sec  [delay '10ms'  '5ms' distribution normal]]
    [15sec  [delay '30ms'  '15ms' distribution normal]]
    [45sec  [delay '50ms'  '25ms' distribution normal]]
    [15sec  [delay '30ms'  '15ms' distribution normal]]
    [15sec  [delay '10ms'  '5ms' distribution normal]]
]

const STEPS_SHALLOW = [
    [time opts];
    [45sec  [delay '15ms' '10ms' distribution normal rate 1mbit ]]
]

# Using Linux netem functionality to manipulate network latency/packet loss
#
# See also:
# https://stackoverflow.com/a/615757
# https://serverfault.com/questions/787006/how-to-add-latency-and-bandwidth-limit-interface-using-tc
export def main [] {
    run-rules $STEPS_SHALLOW 15sec
}
