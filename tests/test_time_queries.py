from datetime import datetime
from zoneinfo import ZoneInfo

from navine.realtime.context import (
    _datetime_in_timezone,
    answer_time_date_query,
    get_local_now,
)


def test_newyork_no_space():
    reply = answer_time_date_query("what time is it in newyork?")
    assert reply is not None
    assert "your local time" not in reply.lower()
    assert "New York" in reply
    ny_now = _datetime_in_timezone("America/New_York")
    ny_clock = ny_now.strftime("%I:%M %p").lstrip("0")
    assert ny_clock in reply


def test_new_york_with_space():
    reply = answer_time_date_query("what time is it in new york?")
    assert reply is not None
    assert "your local time" not in reply.lower()
    assert "New York" in reply


def test_for_me_uses_local():
    reply = answer_time_date_query("what time is it for me?")
    assert reply is not None
    assert "your local time" in reply.lower()
    local_clock = get_local_now().strftime("%I:%M %p").lstrip("0")
    assert local_clock in reply


def test_tokyo():
    reply = answer_time_date_query("what time is it in tokyo?")
    assert reply is not None
    assert "your local time" not in reply.lower()
    assert "Tokyo" in reply
    tokyo_now = _datetime_in_timezone("Asia/Tokyo")
    tokyo_clock = tokyo_now.strftime("%I:%M %p").lstrip("0")
    assert tokyo_clock in reply


def test_nyc_alias():
    reply = answer_time_date_query("what time is it in nyc?")
    assert reply is not None
    assert "New York" in reply


def test_ny_alias():
    reply = answer_time_date_query("what time is it in NY?")
    assert reply is not None
    assert "New York" in reply


def test_unknown_city_not_local():
    reply = answer_time_date_query("what time is it in foobarville?")
    assert reply is not None
    assert "couldn't resolve the timezone" in reply.lower()
    assert "your local time" not in reply.lower()


def test_new_york_differs_from_local_when_zones_differ():
    local = get_local_now()
    ny = datetime.now(ZoneInfo("America/New_York"))
    reply = answer_time_date_query("what time is it in newyork?")
    local_reply = answer_time_date_query("what time is it for me?")
    assert reply is not None
    assert local_reply is not None
    if local.utcoffset() != ny.utcoffset():
        assert reply != local_reply
