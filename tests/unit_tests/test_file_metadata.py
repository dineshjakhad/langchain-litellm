"""File metadata must survive LangChain message normalization."""

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.messages import HumanMessage

from langchain_litellm import ChatLiteLLM
from langchain_litellm.chat_models.litellm import _convert_message_to_dict


@pytest.fixture(params=["remote", "base64"])
def file_block(request: pytest.FixtureRequest) -> dict:
    payload = (
        {"file_id": "gs://bucket/clip.webm"}
        if request.param == "remote"
        else {
            "file_data": "data:application/pdf;base64,JVBERi0=",
            "filename": "document.pdf",
        }
    )
    payload.update(
        format="audio/webm" if request.param == "remote" else "application/pdf",
        video_metadata={"fps": 2, "start_offset": "0s", "end_offset": "1s"},
        custom_option={"enabled": False},
        file_custom="keep-prefix",
    )
    return {"type": "file", "file": payload}


@pytest.mark.parametrize("normalize", [False, True])
def test_file_metadata_conversion(file_block: dict, normalize: bool) -> None:
    message = HumanMessage(content=[file_block])
    if normalize:
        message = _normalize_messages([message])[0]
    original = deepcopy(message.content)

    assert _convert_message_to_dict(message)["content"] == [file_block]
    assert message.content == original


def test_file_metadata_ignores_unrelated_extras() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "file_id": "file-123",
                "extras": {"trace_id": "private", "file_format": "audio/webm"},
            },
            {"type": "file", "file_id": "file-456"},
            {
                "type": "image",
                "url": "https://example.com/image.png",
                "extras": {"file_format": "ignored"},
            },
        ]
    )
    assert _convert_message_to_dict(message)["content"] == [
        {"type": "file", "file": {"file_id": "file-123", "format": "audio/webm"}},
        {"type": "file", "file": {"file_id": "file-456"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


@pytest.mark.parametrize("use_async", [False, True])
async def test_invoke_preserves_file_metadata(
    file_block: dict, use_async: bool
) -> None:
    llm = ChatLiteLLM(model="gemini/gemini-3-flash")
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    }
    method = "acompletion" if use_async else "completion"
    mock_kwargs = {"new_callable": AsyncMock} if use_async else {}
    with patch.object(llm.client, method, return_value=response, **mock_kwargs) as call:
        messages = [HumanMessage(content=[file_block])]
        if use_async:
            await llm.ainvoke(messages)
        else:
            llm.invoke(messages)

    call.assert_called_once()
    assert call.call_args.kwargs["messages"] == [
        {"role": "user", "content": [file_block]}
    ]
