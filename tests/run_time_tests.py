import sys

from navine.realtime.context import (
    _datetime_in_timezone,
    answer_time_date_query,
    get_local_now,
)


def check(name, condition):
    if condition:
        print(f"PASS: {name}")
        return True
    print(f"FAIL: {name}")
    return False


def main():
    passed = 0
    total = 0

    def run(name, condition):
        nonlocal passed, total
        total += 1
        if check(name, condition):
            passed += 1

    reply = answer_time_date_query("what time is it in newyork?")
    ny_now = _datetime_in_timezone("America/New_York")
    ny_clock = ny_now.strftime("%I:%M %p").lstrip("0")
    run("newyork no space", reply is not None and "your local time" not in reply.lower() and "New York" in reply and ny_clock in reply)

    reply = answer_time_date_query("what time is it in new york?")
    run("new york with space", reply is not None and "your local time" not in reply.lower() and "New York" in reply)

    reply = answer_time_date_query("what time is it for me?")
    local_clock = get_local_now().strftime("%I:%M %p").lstrip("0")
    run("for me local", reply is not None and "your local time" in reply.lower() and local_clock in reply)

    reply = answer_time_date_query("what time is it in tokyo?")
    tokyo_now = _datetime_in_timezone("Asia/Tokyo")
    tokyo_clock = tokyo_now.strftime("%I:%M %p").lstrip("0")
    run("tokyo", reply is not None and "your local time" not in reply.lower() and "Tokyo" in reply and tokyo_clock in reply)

    reply = answer_time_date_query("what time is it in nyc?")
    run("nyc alias", reply is not None and "New York" in reply)

    reply = answer_time_date_query("what time is it in NY?")
    run("ny alias", reply is not None and "New York" in reply)

    reply = answer_time_date_query("what time is it in foobarville?")
    run("unknown city", reply is not None and "couldn't resolve the timezone" in reply.lower() and "your local time" not in reply.lower())

    ny_reply = answer_time_date_query("what time is it in newyork?")
    local_reply = answer_time_date_query("what time is it for me?")
    local = get_local_now()
    ny = _datetime_in_timezone("America/New_York")
    if local.utcoffset() != ny.utcoffset():
        run("ny differs from local", ny_reply != local_reply)
    else:
        run("ny differs from local", ny_reply is not None and local_reply is not None)

    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
