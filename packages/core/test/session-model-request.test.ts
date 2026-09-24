import { describe, expect, test } from "bun:test"
import {
  AIError,
  InvalidRequestError,
  LLM,
  LLMEvent,
  type LLMRequest,
  Message,
  ToolResultPart,
  Media,
} from "@opencode/ai"
import { OpenAIChat } from "@opencode/ai/protocols"
import { boundImages, outputLimit, streamAuxiliary, unsupportedParts } from "@opencode/core/session/model-request"
import { Effect, Stream } from "effect"

const capabilities = (input: string[]) => ({ tools: true, input, output: ["text"] })

describe("SessionModelRequest.outputLimit", () => {
  test("requests the catalog output limit up to 128k", () => {
    expect(outputLimit({ context: 1_000_000, output: 128_000 })).toBe(128_000)
    expect(outputLimit({ context: 200_000, output: 64_000 })).toBe(64_000)
    expect(outputLimit({ context: 1_048_576, output: 1_048_576 })).toBe(128_000)
  })

  test("falls back to 32k when the catalog has no output limit", () => {
    expect(outputLimit({ context: 200_000, output: 0 })).toBe(32_000)
  })

  test("fits the limit to the room the prompt leaves in the context window", () => {
    expect(outputLimit({ context: 1_000_000, output: 128_000 }, 50_000)).toBe(128_000)
    expect(outputLimit({ context: 1_000_000, output: 128_000 }, 950_000)).toBe(45_904)
    expect(outputLimit({ context: 128_000, output: 32_000 }, 110_000)).toBe(13_904)
  })

  test("keeps a minimum limit when the prompt nearly fills the context window", () => {
    expect(outputLimit({ context: 200_000, output: 64_000 }, 199_000)).toBe(1_024)
    expect(outputLimit({ context: 200_000, output: 512 }, 199_000)).toBe(512)
  })

  test("ignores the prompt size when the context window is unknown", () => {
    expect(outputLimit({ context: 0, output: 32_000 }, 500_000)).toBe(32_000)
  })
})

describe("SessionModelRequest.streamAuxiliary", () => {
  const request = LLM.request({
    model: OpenAIChat.route.model({ id: "summary-model", provider: "test" }),
    prompt: "Summarize",
    generation: { maxTokens: 32_000, temperature: 0 },
  })
  const rejection = (accepted: number) =>
    new AIError({
      reason: new InvalidRequestError({
        message: `max_tokens: 32000 > ${accepted}, which is the maximum allowed number of output tokens`,
        classification: "output-limit",
        acceptedMaxTokens: accepted,
      }),
    })
  const run = (respond: (request: LLMRequest) => Stream.Stream<LLMEvent, AIError>) => {
    const seen: LLMRequest[] = []
    const events = streamAuxiliary(
      {
        stream: (request) => {
          seen.push(request)
          return respond(request)
        },
      },
      request,
      {},
    ).pipe(Stream.runCollect, Effect.exit, Effect.runSync)
    return { seen, events }
  }

  test("retries once at the output limit the provider accepts", () => {
    const result = run((request) =>
      (request.generation?.maxTokens ?? 0) > 8_000
        ? Stream.fail(rejection(8_000))
        : Stream.make(LLMEvent.textDelta({ id: "text", text: "Summary" })),
    )

    expect(result.seen.map((request) => request.generation)).toEqual([
      request.generation,
      expect.objectContaining({ maxTokens: 8_000, temperature: 0 }),
    ])
    expect(result.events._tag).toBe("Success")
  })

  test("does not retry other failures or a second rejection", () => {
    expect(run(() => Stream.fail(rejection(8_000))).seen).toHaveLength(2)
    const invalid = new AIError({ reason: new InvalidRequestError({ message: "invalid tool schema" }) })
    const result = run(() => Stream.fail(invalid))
    expect(result.seen).toHaveLength(1)
    expect(result.events).toMatchObject({ _tag: "Failure" })
  })
})

describe("SessionModelRequest.unsupportedParts", () => {
  test("replaces unsupported user media with a visible error", () => {
    const messages = unsupportedParts(
      [
        Message.user([
          Message.text("Describe these files"),
          { type: "media", media: Media.base64("aGVsbG8=", "image/png"), filename: "logo.png" },
          { type: "media", media: Media.base64("JVBERg==", "application/pdf"), filename: "document.pdf" },
        ]),
      ],
      capabilities(["text"]),
    )

    expect(messages[0]?.content).toEqual([
      Message.text("Describe these files"),
      Message.text('ERROR: Cannot read "logo.png" (this model does not support image input). Inform the user.'),
      Message.text('ERROR: Cannot read "document.pdf" (this model does not support pdf input). Inform the user.'),
    ])
  })

  test("replaces unsupported media nested in tool results", () => {
    const messages = unsupportedParts(
      [
        Message.tool(
          ToolResultPart.make({
            id: "call_1",
            name: "read",
            result: {
              type: "content",
              value: [
                { type: "text", text: "Image read successfully" },
                { type: "file", uri: "data:image/png;base64,aGVsbG8=", mime: "image/png", name: "logo.png" },
              ],
            },
          }),
        ),
      ],
      capabilities(["text"]),
    )

    expect(messages[0]?.content[0]).toMatchObject({
      type: "tool-result",
      result: {
        type: "content",
        value: [
          { type: "text", text: "Image read successfully" },
          {
            type: "text",
            text: 'ERROR: Cannot read "logo.png" (this model does not support image input). Inform the user.',
          },
        ],
      },
    })
  })

  test("preserves supported media", () => {
    const message = Message.user({ type: "media", media: Media.base64("aGVsbG8=", "image/png") })
    expect(unsupportedParts([message], capabilities(["text", "image"]))[0]?.content).toEqual(message.content)
  })
})

describe("SessionModelRequest.boundImages", () => {
  test("preserves images below the trigger", () => {
    const messages = [Message.user({ type: "media", media: Media.base64("aGVsbG8=", "image/png") })]
    expect(boundImages(messages)).toBe(messages)
  })

  test("replaces oldest images until the retained payload reaches the target", () => {
    const image = "a".repeat(9 * 1024 * 1024)
    const messages = [
      Message.user({ type: "media", media: Media.base64(image, "image/png"), filename: "first.png" }),
      Message.user({ type: "media", media: Media.base64(image, "image/png"), filename: "second.png" }),
      Message.user({ type: "media", media: Media.base64(image, "image/png"), filename: "third.png" }),
    ]
    const result = boundImages(messages)

    expect(result[0]?.content[0]).toMatchObject({ type: "text" })
    expect(result[1]?.content[0]).toMatchObject({ type: "text" })
    expect(result[2]?.content[0]).toMatchObject({ type: "media", filename: "third.png" })
  })

  test("replaces images nested in tool results", () => {
    const image = "a".repeat(13 * 1024 * 1024)
    const result = boundImages([
      Message.tool(
        ToolResultPart.make({
          id: "call_1",
          name: "read",
          result: {
            type: "content",
            value: [
              { type: "file", uri: `data:image/png;base64,${image}`, mime: "image/png", name: "first.png" },
              { type: "file", uri: `data:image/png;base64,${image}`, mime: "image/png", name: "second.png" },
            ],
          },
        }),
      ),
    ])

    expect(result[0]?.content[0]).toMatchObject({
      type: "tool-result",
      result: {
        type: "content",
        value: [{ type: "text" }, { type: "file", name: "second.png" }],
      },
    })
  })
})
