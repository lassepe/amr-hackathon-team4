"""
This macro releases the GIL for the enclosed expression.
Only use this if the wrapped code does *not* call back into Python (even just accessing python data types.)
"""
macro pygil_unlocked(expr)
    quote
        _pythread = PythonCall.C.PyEval_SaveThread()
        try
            $(esc(expr))
        finally
            PythonCall.C.PyEval_RestoreThread(_pythread)
        end
    end
end
